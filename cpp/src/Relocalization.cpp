/*
 * Relocalization.cpp
 */

#include <iomanip>
#include <fstream>
#include <sstream>

#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Relocalization.h"
#include "ApproxKMeans.h"

using namespace std;
using namespace cv;
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
	classifier->findNearest(bow, num_to_return, noArray(), neighborResponses,
			noArray());

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
void Database::setBowExtractor(Ptr<BOWImgDescriptorExtractor> bow_extractor) {
	bowExtractor = bow_extractor;
}

void Database::train() {

	int max_iters = 10;
	TermCriteria terminate_criterion(TermCriteria::MAX_ITER, max_iters, 0.0);
	BOWApproxKMeansTrainer bow_trainer(vocabulary_size, terminate_criterion);

	cout << "computing descriptors for each keyframe..." << endl;
	// iterate through the images
	map<int, vector<KeyPoint>> imageKeypoints;
	map<int, Mat> imageDescriptors;
	for (const auto& element : *frames) {
		const auto& frame = element.second;
		Mat colorImage = imread(frame->imagePath.string());

		imageKeypoints[frame->index] = vector<KeyPoint>();
		imageDescriptors.insert(make_pair(frame->index, Mat()));

		// this can be quite slow, so reload a cached copy from disk if it's available
		if (!frame->loadDescriptors(imageKeypoints[frame->index],
				imageDescriptors[frame->index])) {
			featureDetector->detect(colorImage, imageKeypoints[frame->index]);
			descriptorExtractor->compute(colorImage,
					imageKeypoints[frame->index],
					imageDescriptors[frame->index]);
			frame->saveDescriptors(imageKeypoints[frame->index],
					imageDescriptors[frame->index]);
		}

		if (!imageDescriptors[frame->index].empty()) {
			int descriptor_count = imageDescriptors[frame->index].rows;

			for (int i = 0; i < descriptor_count; i++) {
				bow_trainer.add(imageDescriptors[frame->index].row(i));
			}
		}
	}
	cout << "computed " << bow_trainer.descriptorsCount() << " descriptors in " << frames->size() << " frames."
			<< endl;

	cout << "Training vocabulary..." << endl;

	if (!loadVocabulary(vocabulary)) {
		vocabulary = bow_trainer.cluster();
		saveVocabulary(vocabulary);
	}
	bowExtractor->setVocabulary(vocabulary);
	cout << "Finished training vocabulary." << endl;

	cout << "Computing bow descriptors for each image in training set..." << endl;
	// Create training data by converting each keyframe to a bag of words
	Mat samples((int) frames->size(), vocabulary.rows, CV_32FC1);
	Mat labels((int) frames->size(), 1, CV_32SC1);
	int row = 0;
	for (const auto& element : *frames) {
		const auto& frame = element.second;

		Mat bow;
		bowExtractor->compute(imageDescriptors[frame->index], bow);
		auto cur_row = samples.row(row);
		bow.copyTo(cur_row);
		labels.at<int>(row) = frame->index;

		++row;
	}

	classifier = ml::KNearest::create();
	classifier->train(samples, ml::ROW_SAMPLE, labels);

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
