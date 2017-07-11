/*
 * Relocalization.h
 */

#ifndef SRC_RELOCALIZATION_H_
#define SRC_RELOCALIZATION_H_

#include <Eigen/Core>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"

#include "LargeBagOfWords.h"
#include "MapGenerator.h"

namespace sdl {

template <typename T>
using FileLoader = std::function<T(void)>;

struct Frame {
  int index, dbId;

  FileLoader<cv::Mat> imageLoader;

  boost::filesystem::path pointcloudPath, cachePath, framePath;

  Frame(int index, int db_id) : index(index), dbId(db_id) {}

  void setPath(boost::filesystem::path path) { framePath = path; }

  void setImageLoader(FileLoader<cv::Mat> loader) { imageLoader = loader; }
  void setPointcloudPath(boost::filesystem::path path) {
    pointcloudPath = path;
  }
  void setCachePath(boost::filesystem::path path) { cachePath = path; }

  boost::filesystem::path getDescriptorFilename() const;
  bool loadDescriptors(std::vector<cv::KeyPoint>& keypointsOut,
                       cv::Mat& descriptorsOut) const;
  void saveDescriptors(const std::vector<cv::KeyPoint>& keypoints,
                       const cv::Mat& descriptors) const;

  boost::filesystem::path getSceneCoordinateFilename() const;
  void saveSceneCoordinates(cv::SparseMat coordinate_map) const;
  cv::SparseMat loadSceneCoordinates() const;
};

struct Result {
  explicit Result(Frame& frame) : frame(frame) {}

  Frame& frame;
  std::vector<cv::DMatch> matches;
};

class Database;

class Query {
 public:
  Query(const unsigned int parent_database_id, const Frame* frame);

  void setupFeatureDetector(bool detect_from_depth_map);
  void setDescriptorExtractor(
      cv::Ptr<cv::DescriptorExtractor> descriptor_extractor);
  const cv::Mat readColorImage() const;
  const std::vector<cv::KeyPoint>& getKeypoints() const;
  const cv::Mat computeDescriptors();

  const unsigned int getParentDatabaseId() const { return parent_database_id; }
  const Frame* const getFrame() const { return frame; }

 private:
  unsigned int parent_database_id;
  const Frame* frame;

  bool detectFromDepthMaps;
  cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
  std::vector<cv::KeyPoint> keypoints;
};

class Database {
 public:
  Database();
  Database(Database&&);
  ~Database();

  void addFrame(std::unique_ptr<Frame> frame);
  std::map<int, std::unique_ptr<Frame>>* getFrames() { return frames.get(); }

  std::vector<Result> lookup(Query& query, unsigned int num_to_return);

  void setVocabularySize(int size) { vocabulary_size = size; }
  void setMapper(std::unique_ptr<MapGenerator> map_gen);
  void setupFeatureDetector(bool detect_from_depth_maps);
  void setDescriptorExtractor(
      cv::Ptr<cv::DescriptorExtractor> descriptor_extractor);
  void setBowExtractor(
      cv::Ptr<cv::BOWSparseImgDescriptorExtractor> bow_extractor);
  void train();

  void setCachePath(boost::filesystem::path path) { cachePath = path; }

  const boost::filesystem::path& getCachePath() { return cachePath; }

  void setMapper(MapGenerator* map_gen) {
    mapGen = std::unique_ptr<MapGenerator>(map_gen);
    // set the calibration before we clear this pointer later
    // TODO: clean this up
    K_ = mapGen->getCalibration();
  }
  MapGenerator* getMapper() { return mapGen.get(); }

  cv::Mat getCalibration() {
    return K_;
  }

  unsigned int db_id;

 private:
  boost::filesystem::path getVocabularyFilename() const;
  bool loadVocabulary(cv::Mat& vocabularyOut) const;
  void saveVocabulary(const cv::Mat& vocabulary) const;

  // Utility functions used in train()
  void doMapping();
  bool needToRecomputeSceneCoordinates() const;
  int computeDescriptorsForEachFrame(
      std::map<int, std::vector<cv::KeyPoint>>& image_keypoints,
      std::map<int, cv::Mat>& image_descriptors);
  void doClustering(const std::map<int, cv::Mat>& image_descriptors);
  std::map<int, std::vector<int>> computeBowDescriptors(
      const std::map<int, cv::Mat>& image_descriptors);
  Eigen::MatrixXf generateRandomProjection(int descriptor_size, int num_rows);
  Eigen::MatrixXf computeHammingThresholds(
      const Eigen::MatrixXf& projection_matrix,
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

  std::unique_ptr<std::map<int, std::unique_ptr<Frame>>> frames;

  std::unique_ptr<InvertedIndexImpl> pInvertedIndexImpl;

  cv::Mat K_;
};
}  // namespace sdl

#endif /* SRC_RELOCALIZATION_H_ */
