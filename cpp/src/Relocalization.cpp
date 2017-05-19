/*
 * Relocalization.cpp
 */

#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Relocalization.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

namespace sdl {
Database::Database() : frames(new map<int, unique_ptr<Frame>>()) {
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

	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;
	int vocab_size = 100000;
	BOWKMeansTrainer bow_trainer(vocab_size, terminate_criterion);

	cout << "computing descriptors for each keyframe..." << endl;
	// iterate through the images
	map<int, vector<KeyPoint>> imageKeypoints;
	map<int, Mat> imageDescriptors;
	for (const auto& element : *frames) {
		const auto& frame = element.second;
		Mat colorImage = imread(frame->imagePath.string());

		imageKeypoints[frame->index] = vector<KeyPoint>();
		featureDetector->detect(colorImage, imageKeypoints[frame->index]);
		imageDescriptors.insert(make_pair(frame->index, Mat()));
		descriptorExtractor->compute(colorImage,
				imageKeypoints[frame->index], imageDescriptors[frame->index]);

		if (!imageDescriptors[frame->index].empty()) {
			int descriptor_count = imageDescriptors[frame->index].rows;

			for (int i = 0; i < descriptor_count; i++) {
				bow_trainer.add(imageDescriptors[frame->index].row(i));
			}
		}
	}
	cout << "computed " << bow_trainer.descriptorsCount() << " descriptors."
			<< endl;

	cout << "Training vocabulary..." << endl;
	vocabulary = bow_trainer.cluster();
	bowExtractor->setVocabulary(vocabulary);

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
