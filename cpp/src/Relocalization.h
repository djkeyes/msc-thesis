/*
 * Relocalization.h
 */

#ifndef SRC_RELOCALIZATION_H_
#define SRC_RELOCALIZATION_H_

#include <Eigen/Core>
#include <functional>
#include <list>
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
  bool descriptorsExist() const;
  int countDescriptors() const;
  void saveDescriptors(const std::vector<cv::KeyPoint>& keypoints,
                       const cv::Mat& descriptors) const;

  boost::filesystem::path getSceneCoordinateFilename() const;
  void saveSceneCoordinates(const SceneCoordinateMap& coordinate_map) const;
  SceneCoordinateMap loadSceneCoordinates() const;
  boost::filesystem::path getPoseFilename() const;
  void savePose(const dso::SE3& pose) const;
  dso::SE3 loadPose() const;
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

  std::vector<Result> lookup(
      Query& query, unsigned int num_to_return,
      std::vector<std::reference_wrapper<Database>>& all_training_dbs);

  void setVocabularySize(int size) { vocabulary_size = size; }
  void setMapper(std::unique_ptr<DsoMapGenerator> map_gen);
  void setupFeatureDetector(bool detect_from_depth_maps);
  void setDescriptorExtractor(
      cv::Ptr<cv::DescriptorExtractor> descriptor_extractor);
  void setBowExtractor(
      cv::Ptr<cv::BOWSparseImgDescriptorExtractor> bow_extractor);
  void runVoAndComputeDescriptors();
  void assignAndBuildIndex(
      std::vector<std::reference_wrapper<Database>>& all_training_dbs);

  void setTrainOnFirstHalf(bool train_on_first_half) {
    trainOnFirstHalf = train_on_first_half;
  }

  void setCachePath(boost::filesystem::path path) { cachePath = path; }

  const boost::filesystem::path& getCachePath() { return cachePath; }

  void setMapper(DsoMapGenerator* map_gen) {
    mapGen = std::unique_ptr<DsoMapGenerator>(map_gen);
    // set the calibration before we clear this pointer later
    // TODO: clean this up
    K_ = mapGen->getCalibration();
  }
  void setGtPoseLoader(std::function<bool(const Frame&, cv::Mat&, cv::Mat&)> gt_pose_loader) {
    gtPoseLoader = gt_pose_loader;
  }
  DsoMapGenerator* getMapper() { return mapGen.get(); }

  cv::Mat getCalibration() {
    return K_;
  }

  void copyVocabularyFileFrom(const Database& from) const;
  bool hasCachedVocabularyFile() const;

  void onlyDoMapping();

  unsigned int db_id;

 private:
  bool trainOnFirstHalf;

  boost::filesystem::path getVocabularyFilename() const;
  bool loadVocabulary(cv::Mat& vocabularyOut) const;
  void saveVocabulary(const cv::Mat& vocabulary) const;

  // Utility functions used in train()
  void doMapping();
  bool needToRecomputeSceneCoordinates() const;
  bool hasCachedDescriptors() const;
  int computeDescriptorsForEachFrame(
      std::map<int, std::vector<cv::KeyPoint>>& image_keypoints,
      std::map<int, cv::Mat>& image_descriptors);
  int loadAllDescriptors(
      std::map<int, std::map<int, std::vector<cv::KeyPoint>>>& image_keypoints,
      std::map<int, std::map<int, cv::Mat>>& image_descriptors,
      std::vector<std::reference_wrapper<Database>>& all_training_dbs);
  boost::filesystem::path mergedDescriptorsFileFlag();
  bool hasCoobservedDescriptors();
  void mergeCoobservedDescriptors(
      const std::map<int, std::vector<cv::KeyPoint>>& image_keypoints,
      const std::map<int, cv::Mat>& image_descriptors);
  void doClustering(
      const std::map<int, std::map<int, cv::Mat>>& image_descriptors,
      std::vector<std::reference_wrapper<Database>>& all_train_dbs);
  std::map<int, std::vector<int>> computeBowDescriptors(
      const std::map<int, cv::Mat>& image_descriptors);
  Eigen::MatrixXf generateRandomProjection(int num_rows);
  Eigen::MatrixXf computeHammingThresholds(
      const Eigen::MatrixXf& projection_matrix,
      const std::map<int, std::map<int, cv::Mat>>& image_descriptors,
      const std::map<int, std::map<int, std::vector<int>>>
          descriptor_assignments,
      std::vector<std::reference_wrapper<Database>>& all_training_dbs);
  struct InvertedIndexImpl;
  boost::filesystem::path getInvertedIndexFilename() const;
  boost::filesystem::path getInvertedIndexWeightsFilename() const;
  bool loadInvertedIndex(InvertedIndexImpl& inverted_index_impl) const;
  void saveInvertedIndex(const InvertedIndexImpl& inverted_index_impl) const;
  void buildInvertedIndex(
      const std::map<int, std::map<int, std::vector<cv::KeyPoint>>>&
          image_keypoints,
      const std::map<int, std::map<int, cv::Mat>>& image_descriptors,
      std::vector<std::reference_wrapper<Database>>& all_training_dbs);

  boost::filesystem::path cachePath;

  int vocabulary_size = 100000;

  std::unique_ptr<DsoMapGenerator> mapGen;

  cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
  bool associateWithDepthMaps;

  cv::Mat vocabulary;
  cv::Ptr<cv::BOWSparseImgDescriptorExtractor> bowExtractor;

  std::unique_ptr<std::map<int, std::unique_ptr<Frame>>> frames;

  std::unique_ptr<InvertedIndexImpl> pInvertedIndexImpl;

  std::function<bool(const Frame&, cv::Mat&, cv::Mat&)> gtPoseLoader;

  cv::Mat K_;
};

std::map<int, std::list<cv::DMatch>> doRatioTest(
    cv::Mat query_descriptors, cv::Mat db_descriptors,
    const std::vector<cv::KeyPoint>& query_keypoints,
    const std::vector<cv::KeyPoint>& db_keypoints,
    cv::Ptr<cv::DescriptorMatcher> matcher, double ratio_threshold,
    bool use_distance_threshold);
}  // namespace sdl

#endif /* SRC_RELOCALIZATION_H_ */
