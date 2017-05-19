/*
 * Relocalization.h
 */

#ifndef SRC_RELOCALIZATION_H_
#define SRC_RELOCALIZATION_H_

#include <string>
#include <vector>

#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "boost/filesystem.hpp"

namespace sdl {
struct Frame {
	int index;
	boost::filesystem::path depthmapPath, imagePath, pointcloudPath;
	Frame(int index) :
			index(index) {
	}

	void setDepthmapPath(boost::filesystem::path path) {
		depthmapPath = path;
	}
	void setImagePath(boost::filesystem::path path) {
		imagePath = path;
	}
	void setPointcloudPath(boost::filesystem::path path) {
		pointcloudPath = path;
	}
};

struct Result {
	Result(Frame& frame) :
			frame(frame) {
	}

	Frame& frame;
};

class Database;

class Query {
public:
	Query(const Database * const parent_database, const Frame * const frame);

	void computeFeatures();
	void setFeatureDetector(cv::Ptr<cv::FeatureDetector> feature_detector);
	const cv::Mat& getColorImage() const;
	const std::vector<cv::KeyPoint>& getKeypoints() const;

	const Database * const parent_database;
private:
	const Frame * const frame;

	cv::Ptr<cv::FeatureDetector> featureDetector;
	std::vector<cv::KeyPoint> keypoints;

};

class Database {
public:
	Database();
	void addFrame(std::unique_ptr<Frame> frame);

	std::vector<Result> lookup(const Query& query, int num_to_return);

	void setFeatureDetector(cv::Ptr<cv::FeatureDetector> feature_detector);
	void setDescriptorExtractor(
			cv::Ptr<cv::DescriptorExtractor> descriptor_extractor);
	void setBowExtractor(cv::Ptr<cv::BOWImgDescriptorExtractor> bow_extractor);
	void train();

private:
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	cv::Ptr<cv::FeatureDetector> featureDetector;

	cv::Mat vocabulary;
	cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor;
	cv::Ptr<cv::ml::KNearest> classifier;

	std::unique_ptr<std::map<int, std::unique_ptr<Frame>>> frames;

};

}

#endif /* SRC_RELOCALIZATION_H_ */
