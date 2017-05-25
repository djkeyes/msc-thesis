/*
 * Relocalization.cpp
 */

#include <iomanip>
#include <fstream>
#include <sstream>
#include <Eigen/QR>

#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "geometricburstiness/inverted_index.h"

#include "Relocalization.h"
#include "LargeBagOfWords.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace fs = boost::filesystem;

namespace sdl {

fs::path Frame::getDescriptorFilename() const {
	stringstream ss;
	ss << "frame_" << setfill('0') << setw(6) << index
			<< "_keypoints_and_descriptors.bin";
	fs::path filename = cachePath / ss.str();
	return filename;
}

bool Frame::loadDescriptors(vector<KeyPoint>& keypointsOut,
		Mat& descriptorsOut) const {
	if (cachePath.empty()) {
		return false;
	}

	fs::path filename(getDescriptorFilename());

	if (!fs::exists(filename)) {
		return false;
	}

	ifstream ifs(filename.string(), ios_base::in | ios_base::binary);

	int num_descriptors, descriptor_size, data_type;
	ifs.read((char*) &num_descriptors, sizeof(u_int32_t));
	ifs.read((char*) &descriptor_size, sizeof(u_int32_t));
	ifs.read((char*) &data_type, sizeof(u_int32_t));

	descriptorsOut.create(num_descriptors, descriptor_size, data_type);
	ifs.read((char*) descriptorsOut.data,
			num_descriptors * descriptor_size * descriptorsOut.elemSize());

	keypointsOut.reserve(num_descriptors);
	for (int i = 0; i < num_descriptors; i++) {
		float x, y, size, angle, response;
		int octave, class_id;
		ifs.read((char*) &x, sizeof(float));
		ifs.read((char*) &y, sizeof(float));
		ifs.read((char*) &size, sizeof(float));
		ifs.read((char*) &angle, sizeof(float));
		ifs.read((char*) &response, sizeof(float));
		ifs.read((char*) &octave, sizeof(uint32_t));
		ifs.read((char*) &class_id, sizeof(uint32_t));
		keypointsOut.emplace_back(x, y, size, angle, response, octave,
				class_id);
	}

	ifs.close();

	return true;
}
void Frame::saveDescriptors(const vector<KeyPoint>& keypoints,
		const Mat& descriptors) const {
	if (cachePath.empty()) {
		return;
	}

	fs::path filename(getDescriptorFilename());
	// create directory if it doesn't exist
	fs::create_directories(filename.parent_path());

	ofstream ofs(filename.string(), ios_base::out | ios_base::binary);
	int descriptor_size = descriptors.size().width;
	int num_descriptors = descriptors.size().height;
	int data_type = descriptors.type();

	ofs.write((char*) &num_descriptors, sizeof(uint32_t));
	ofs.write((char*) &descriptor_size, sizeof(uint32_t));
	ofs.write((char*) &data_type, sizeof(uint32_t));

	ofs.write((char*) descriptors.data,
			num_descriptors * descriptor_size * descriptors.elemSize());
	for (int i = 0; i < num_descriptors; i++) {
		ofs.write((char*) &keypoints[i].pt.x, sizeof(float));
		ofs.write((char*) &keypoints[i].pt.y, sizeof(float));
		ofs.write((char*) &keypoints[i].size, sizeof(float));
		ofs.write((char*) &keypoints[i].angle, sizeof(float));
		ofs.write((char*) &keypoints[i].response, sizeof(float));
		ofs.write((char*) &keypoints[i].octave, sizeof(uint32_t));
		ofs.write((char*) &keypoints[i].class_id, sizeof(uint32_t));
	}

	ofs.close();
}

Database::Database() :
		frames(new map<int, unique_ptr<Frame>>()) {
}

vector<Result> Database::lookup(const Query& query, int num_to_return) {

	Mat bow;
	// apparently compute() may modify the keypoints, because it invokes Feature2D.compute()
	// therefore, store the result locally instead of caching them in the query.
//		vector<KeyPoint> keypoints;
//		bowExtractor->compute(query.getColorImage(), keypoints, bow);

	bowExtractor->compute(query.getColorImage(),
			const_cast<vector<KeyPoint>&>(query.getKeypoints()), bow);

	Mat neighborResponses;

	vector<Result> results;
	results.reserve(num_to_return);
	for (int i = 0; i < num_to_return; i++) {
		// KNearest casts labels to a float, despite the fact that we only ever passed it ints
		int neighborIdx = static_cast<int>(neighborResponses.at<float>(i));
		Frame& keyframe = *frames->at(neighborIdx);
		Result cur_result(keyframe);

		results.push_back(cur_result);
	}

	return results;
}

void Database::setFeatureDetector(Ptr<FeatureDetector> feature_detector) {
	featureDetector = feature_detector;
}

void Database::setDescriptorExtractor(
		Ptr<DescriptorExtractor> descriptor_extractor) {
	descriptorExtractor = descriptor_extractor;
}
void Database::setBowExtractor(
		Ptr<BOWSparseImgDescriptorExtractor> bow_extractor) {
	bowExtractor = bow_extractor;
}

int Database::computeFrameDescriptors(
		map<int, vector<KeyPoint>>& image_keypoints,
		map<int, Mat>& image_descriptors) {

	int total_descriptors = 0;

	// iterate through the images
	for (const auto& element : *frames) {
		const auto& frame = element.second;
		Mat colorImage = imread(frame->imagePath.string());

		image_keypoints[frame->index] = vector<KeyPoint>();
		image_descriptors.insert(make_pair(frame->index, Mat()));

		// this can be quite slow, so reload a cached copy from disk if it's available
		if (!frame->loadDescriptors(image_keypoints[frame->index],
				image_descriptors[frame->index])) {
			featureDetector->detect(colorImage, image_keypoints[frame->index]);
			descriptorExtractor->compute(colorImage,
					image_keypoints[frame->index],
					image_descriptors[frame->index]);
			frame->saveDescriptors(image_keypoints[frame->index],
					image_descriptors[frame->index]);
		}
		total_descriptors += image_descriptors[frame->index].rows;
	}
	return total_descriptors;
}
void Database::doClustering(const map<int, Mat>& image_descriptors) {

	if (!loadVocabulary(vocabulary)) {
		int max_iters = 10;
		TermCriteria terminate_criterion(TermCriteria::MAX_ITER, max_iters,
				0.0);
		BOWApproxKMeansTrainer bow_trainer(vocabulary_size,
				terminate_criterion);

		for (const auto& element : *frames) {
			int index = element.second->index;
			if (!image_descriptors.at(index).empty()) {
				int descriptor_count = image_descriptors.at(index).rows;

				for (int i = 0; i < descriptor_count; i++) {
					bow_trainer.add(image_descriptors.at(index).row(i));
				}
			}
		}
		vocabulary = bow_trainer.cluster();
		saveVocabulary(vocabulary);
	}
	bowExtractor->setVocabulary(vocabulary);
}

map<int, vector<int>> Database::computeBowDescriptors(
		const map<int, Mat>& image_descriptors) {
	// Create training data by converting each keyframe to a bag of words
	int dims[] = { static_cast<int>(frames->size()), vocabulary.rows };
	SparseMat samples(2, dims, CV_32FC1);
	map<int, vector<int>> assignments;
	for (const auto& element : *frames) {
		int index = element.second->index;

		bowExtractor->computeAssignments(image_descriptors.at(index),
				assignments[index]);
	}
	return assignments;
}

MatrixXf Database::generateRandomProjection(int descriptor_size, int num_rows) {
	default_random_engine generator;
	normal_distribution<float> distribution(0.0, 1.0);

	MatrixXf random_matrix(descriptor_size, descriptor_size);
	for (int i = 0; i < descriptor_size; i++) {
		for (int j = 0; j < descriptor_size; j++) {
			random_matrix(i, j) = distribution(generator);
		}
	}
	ColPivHouseholderQR<MatrixXf> qr_decomp(random_matrix);
	MatrixXf Q = qr_decomp.householderQ();
	MatrixXf projection = Q.topRows(num_rows);
	return projection;
}
MatrixXf Database::computeHammingThresholds(const MatrixXf& projection_matrix,
		const map<int, Mat>& image_descriptors,
		const map<int, vector<int>> descriptor_assignments) {
	Matrix<float, 64, Eigen::Dynamic> hamming_thresholds;
	hamming_thresholds.resize(64, vocabulary_size);
	int descriptor_size = descriptorExtractor->descriptorSize();

	int num_images = static_cast<int>(image_descriptors.size());
	cout << " Found " << num_images << " database images " << endl;

	// Loads for each word up to 10k nearby descriptors and then computes the thresholds.
	vector<vector<vector<float> > > entries_per_word(vocabulary_size);
	for (int i = 0; i < vocabulary_size; ++i) {
		entries_per_word[i].resize(64);
		for (int j = 0; j < 64; ++j)
			entries_per_word[i][j].clear();
	}
	vector<int> num_desc_per_word(vocabulary_size, 0);
	int num_missing_words = vocabulary_size;

	vector<int> randomly_permuted_db_ids(num_images);
	for (int i = 0; i < num_images; ++i) {
		randomly_permuted_db_ids[i] = i;
	}
	random_shuffle(randomly_permuted_db_ids.begin(),
			randomly_permuted_db_ids.end());

	cout << " Determining relevant images per word " << endl;
	const int kNumDesiredDesc = 10000;
	for (int i = 0; i < num_images; ++i) {
		if (num_missing_words == 0)
			break;

		int id = randomly_permuted_db_ids[i];

		const Mat descriptors = image_descriptors.at(id);

		int num_features = descriptors.rows;

		if (num_features == 0) {
			continue;
		}

		for (int j = 0; j < num_features; ++j) {
			const int closest_word = descriptor_assignments.at(id)[j];
			if (num_desc_per_word[closest_word] >= kNumDesiredDesc) {
				continue;
			}

			// map the opencv memory to an eigen matrix
			// Ostensibly the opencv matrix is 1xD, and the eigen matrix needs
			// to be Dx1, but it doesn't really matter since it's 1-dimensional
			// and stored contiguously.
			Map<MatrixXf> descriptor((float*)descriptors.row(j).data, descriptor_size, 1);

			Eigen::Matrix<float, 64, 1> proj_sift = projection_matrix
					* descriptor;

			for (int k = 0; k < 64; ++k) {
				entries_per_word[closest_word][k].push_back(proj_sift[k]);
			}
			num_desc_per_word[closest_word] += 1;

			if (num_desc_per_word[closest_word] == kNumDesiredDesc) {
				--num_missing_words;
			}
		}

		if (i % 100 == 0) {
			cout << "\r " << i << flush;
		}
	}
	cout << endl;

	// For each word, computes the thresholds.
	cout << " Computing the thresholds per word " << endl;
	for (int i = 0; i < vocabulary_size; ++i) {
		int num_desc = num_desc_per_word[i];

		if (num_desc == 0) {
			cout << " WARNING: FOUND EMPTY WORD " << i << endl;
			hamming_thresholds.col(i) = Eigen::Matrix<float, 64, 1>::Zero();
		} else {
			const int median_element = num_desc / 2;
			for (int k = 0; k < 64; ++k) {
				nth_element(entries_per_word[i][k].begin(),
						entries_per_word[i][k].begin() + median_element,
						entries_per_word[i][k].end());
				hamming_thresholds(k, i) =
						entries_per_word[i][k][median_element];
			}
		}

		if (i % 1000 == 0) {
			cout << "\r word " << i << flush;
		}
	}
	cout << " done" << endl;

	return hamming_thresholds;
}
void Database::buildInvertedIndex(
		const map<int, vector<KeyPoint>>& image_keypoints,
		const map<int, Mat>& image_descriptors,
		const map<int, vector<int>> descriptor_assignments) {

	// TODO: load an index if it exists, and cache a copy if it didn't exist. We might also need to save the random projection, not sure if InvertedIndex saves that by default.

	MatrixXf projection_matrix = generateRandomProjection(
			descriptorExtractor->descriptorSize(), 64);
	MatrixXf hamming_thresholds = computeHammingThresholds(projection_matrix,
			image_descriptors, descriptor_assignments);

	geometric_burstiness::InvertedIndex<64> inverted_index;
	cout << "initializing index" << endl;
	inverted_index.InitializeIndex(vocabulary_size);
	cout << "finished initialization" << endl;

	inverted_index.SetProjectionMatrix(projection_matrix);
	inverted_index.SetHammingThresholds(hamming_thresholds);

	int descriptor_size = descriptorExtractor->descriptorSize();

	int total_number_entries = 0;
	for (const auto& element : *frames) {
		int index = element.second->index;
		const vector<cv::KeyPoint>& keypoints = image_keypoints.at(index);
		const Mat& descriptors = image_descriptors.at(index);

		int num_features = static_cast<int>(keypoints.size());

		if (num_features == 0) {
			continue;
		}

		for (int j = 0; j < num_features; ++j, ++total_number_entries) {
			geometric_burstiness::InvertedFileEntry<64> entry;
			entry.image_id = index;
			entry.x = keypoints[j].pt.x;
			entry.y = keypoints[j].pt.y;
			// TODO(daniel): These are geometric properties of affine
			// descriptors. I think these are only used during geometric-based
			// reranking, so maybe it's okay to leave them empty. For regular
			// SIFT, it might be possible to compute these values, but I don't
			// see how this would be computable for a general descriptor.
			entry.a = 0;
			entry.b = 0;
			entry.c = 0;

			const int closest_word = descriptor_assignments.at(index)[j];
			if (closest_word < 0 || closest_word >= vocabulary_size) {
				throw runtime_error("Impossible word " + closest_word);
			}

			Map<MatrixXf> descriptor((float*)descriptors.row(j).data, descriptor_size, 1);

			inverted_index.AddEntry(entry, closest_word, descriptor);
		}
	}
	cout << " The index contains " << total_number_entries << " entries "
			<< "in total" << endl;

	cout << " Estimates the descriptor space densities " << endl;
	inverted_index.EstimateDescriptorSpaceDensities();

	inverted_index.FinalizeIndex();
	cout << " Inverted index finalized" << endl;

}
void Database::train() {

	map<int, vector<KeyPoint>> image_keypoints;
	map<int, Mat> image_descriptors;

	cout << "computing descriptors for each keyframe..." << endl;
	int descriptor_count = computeFrameDescriptors(image_keypoints,
			image_descriptors);
	cout << "computed " << descriptor_count << " descriptors in "
			<< frames->size() << " frames." << endl;

	cout << "Training vocabulary..." << endl;
	doClustering(image_descriptors);
	cout << "Finished training vocabulary." << endl;

	cout << "Computing bow descriptors for each image in training set using nearest neighbor to each descriptor..." << endl;
	map<int, vector<int>> descriptor_assignments = computeBowDescriptors(image_descriptors);
	cout << "Finished computing bags of words." << endl;

	cout << "Building inverted index..." << endl;
	buildInvertedIndex(image_keypoints, image_descriptors, descriptor_assignments);
	cout << "Finished inverted index." << endl;
}

void Database::addFrame(unique_ptr<Frame> frame) {
	int index = frame->index;
	frames->insert(make_pair(index, move(frame)));
}

fs::path Database::getVocabularyFilename() const {
	return cachePath / "clusters.bin";
}
bool Database::loadVocabulary(cv::Mat& vocabularyOut) const {

	if (cachePath.empty()) {
		return false;
	}

	fs::path filename(getVocabularyFilename());

	if (!fs::exists(filename)) {
		return false;
	}

	ifstream ifs(filename.string(), ios_base::in | ios_base::binary);

	int rows, cols, size, type;
	ifs.read((char*) &rows, sizeof(uint32_t));
	ifs.read((char*) &cols, sizeof(uint32_t));
	ifs.read((char*) &size, sizeof(uint32_t));
	ifs.read((char*) &type, sizeof(uint32_t));
	vocabularyOut.create(rows, cols, type);
	ifs.read((char*) vocabularyOut.data, rows * cols * size);

	ifs.close();

	return true;
}
void Database::saveVocabulary(const cv::Mat& vocabulary) const {
	if (cachePath.empty()) {
		return;
	}

	fs::path filename(getVocabularyFilename());
	// create directory if it doesn't exist
	fs::create_directories(filename.parent_path());

	ofstream ofs(filename.string(), ios_base::out | ios_base::binary);

	int size = vocabulary.elemSize();
	int type = vocabulary.type();
	ofs.write((char*) &vocabulary.rows, sizeof(uint32_t));
	ofs.write((char*) &vocabulary.cols, sizeof(uint32_t));
	ofs.write((char*) &size, sizeof(uint32_t));
	ofs.write((char*) &type, sizeof(uint32_t));
	ofs.write((char*) vocabulary.data,
			vocabulary.rows * vocabulary.cols * size);

	ofs.close();
}

Query::Query(const Database * const parent_database, const Frame * const frame) :
		parent_database(parent_database), frame(frame) {
}

void Query::computeFeatures() {
	// TODO a short sequence contains many frames. We should pick one
	// from the middle or something. Or instead of computing many short
	// sequences, we should just use the whole sequence (but only consider
	// frames in a window around the current keyframe).
//	CV::Mat colorImage(imread(pickAnImage(dataDir.string())))
//	featureDetector->detect(colorImage, keypoints);
}
void Query::setFeatureDetector(Ptr<FeatureDetector> feature_detector) {
	featureDetector = feature_detector;
}

const Mat& Query::getColorImage() const {
	throw runtime_error("not implemented");
}
const vector<KeyPoint>& Query::getKeypoints() const {
	return keypoints;
}

}
