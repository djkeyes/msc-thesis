/*
 * Relocalization.h
 */

#ifndef SRC_RELOCALIZATION_H_
#define SRC_RELOCALIZATION_H_

#include <string>
#include <vector>
#include <Eigen/Core>

#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "boost/filesystem.hpp"

#include "MapGenerator.h"
#include "LargeBagOfWords.h"

namespace sdl {
struct Frame {
	int index;
	boost::filesystem::path depthmapPath, imagePath, pointcloudPath, cachePath;

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
	void setCachePath(boost::filesystem::path path) {
		cachePath = path;
	}

	boost::filesystem::path getDescriptorFilename() const;
	bool loadDescriptors(std::vector<cv::KeyPoint>& keypointsOut,
			cv::Mat& descriptorsOut) const;
	void saveDescriptors(const std::vector<cv::KeyPoint>& keypoints,
			const cv::Mat& descriptors) const;

};

struct Result {
	Result(Frame& frame) :
			frame(frame) {
	}

	Frame& frame;
	std::vector<cv::DMatch> matches;
};

class Database;

class Query {
public:
	Query(const unsigned int parent_database_id, const Frame * const frame);

	void computeFeatures();
	void setupFeatureDetector(bool detect_from_depth_map);
	void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor> descriptor_extractor);
	const cv::Mat readColorImage() const;
	const std::vector<cv::KeyPoint>& getKeypoints() const;
	const cv::Mat& getDescriptors() const;

	const unsigned int parent_database_id;
	const Frame * const frame;
private:


	bool detectFromDepthMaps;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

};

class Database {
public:
	Database();
	Database(Database&&);
	~Database();

	void addFrame(std::unique_ptr<Frame> frame);

	std::vector<Result> lookup(const Query& query, unsigned int num_to_return);

	void setVocabularySize(int size) {
		vocabulary_size = size;
	}
	void setMapper(std::unique_ptr<MapGenerator> map_gen);
	void setupFeatureDetector(bool detect_from_depth_maps);
	void setDescriptorExtractor(
			cv::Ptr<cv::DescriptorExtractor> descriptor_extractor);
	void setBowExtractor(
			cv::Ptr<cv::BOWSparseImgDescriptorExtractor> bow_extractor);
	void train();

	void setCachePath(boost::filesystem::path path) {
		cachePath = path;
	}

	const boost::filesystem::path& getCachePath() {
		return cachePath;
	}

	void setMapper(MapGenerator* map_gen) {
		mapGen = std::unique_ptr<MapGenerator>(map_gen);
	}

	unsigned int db_id;

private:

	boost::filesystem::path getVocabularyFilename() const;
	bool loadVocabulary(cv::Mat& vocabularyOut) const;
	void saveVocabulary(const cv::Mat& vocabulary) const;

	// Utility functions used in train()
	void doMapping();
	boost::filesystem::path getSceneCoordinateFilename(int frame_id) const;
	bool needToRecomputeSceneCoordinates() const;
	void saveSceneCoordinates(int frame_id, cv::SparseMat coordinate_map) const;
	cv::SparseMat loadSceneCoordinates(int frame_id) const;
	int computeDescriptorsForEachFrame(
			std::map<int, std::vector<cv::KeyPoint>>& image_keypoints,
			std::map<int, cv::Mat>& image_descriptors);
	void doClustering(const std::map<int, cv::Mat>& image_descriptors);
	std::map<int, std::vector<int>> computeBowDescriptors(const std::map<int, cv::Mat>& image_descriptors);
	Eigen::MatrixXf generateRandomProjection(int descriptor_size, int num_rows);
	Eigen::MatrixXf computeHammingThresholds(const Eigen::MatrixXf& projection_matrix,
			const std::map<int, cv::Mat>& image_descriptors,
			const std::map<int, std::vector<int>> descriptor_assignments);
	struct InvertedIndexImpl;
	boost::filesystem::path getInvertedIndexFilename() const;
	boost::filesystem::path getInvertedIndexWeightsFilename() const;
	bool loadInvertedIndex(InvertedIndexImpl& inverted_index_impl) const;
	void saveInvertedIndex(const InvertedIndexImpl& inverted_index_impl) const;
	void buildInvertedIndex(
			const std::map<int, std::vector<cv::KeyPoint>>& image_keypoints,
			const std::map<int, cv::Mat>& image_descriptors);

	boost::filesystem::path cachePath;

	int vocabulary_size = 100000;

	std::unique_ptr<MapGenerator> mapGen;

	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	bool associateWithDepthMaps;

	cv::Mat vocabulary;
	cv::Ptr<cv::BOWSparseImgDescriptorExtractor> bowExtractor;

	std::unique_ptr<std::map<int, std::unique_ptr<Frame>>>frames;

	std::unique_ptr<InvertedIndexImpl> pInvertedIndexImpl;
};

}

#endif /* SRC_RELOCALIZATION_H_ */
