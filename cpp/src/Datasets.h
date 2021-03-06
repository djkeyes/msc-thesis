
#ifndef SRC_DATASETS_H_
#define SRC_DATASETS_H_

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <tuple>
#include <utility>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "boost/filesystem/path.hpp"

#include "Relocalization.h"

namespace sdl {

inline std::tuple<cv::Mat, int, int> getDummyCalibration(cv::Mat probe_image) {
  int img_width, img_height;
  img_width = probe_image.cols;
  img_height = probe_image.rows;

  cv::Mat K = cv::Mat::zeros(3, 3, CV_32FC1);
  K.at<float>(0, 0) = (img_width + img_height) / 2.;
  K.at<float>(1, 1) = (img_width + img_height) / 2.;
  K.at<float>(0, 2) = img_width / 2 - 0.5;
  K.at<float>(1, 2) = img_height / 2 - 0.5;
  K.at<float>(2, 2) = 1;
  return std::make_tuple(K, img_width, img_height);
}

class SceneParser {
 public:
  virtual ~SceneParser() {}

  /*
   * Parse a scene into a set of databases and a set of queries (which may be
   * built from overlapping data).
   */
  virtual void parseScene(
      std::vector<sdl::Database>& dbs,
      std::vector<std::pair<std::vector<std::reference_wrapper<sdl::Database>>,
                            std::vector<sdl::Query>>>& dbs_with_queries,
      bool inject_ground_truth) = 0;

  /*
   * Given a frame, load the ground truth pose (as a rotation and translation
   * matrix) from the dataset.
   */
  virtual bool loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                                   cv::Mat& translation) = 0;
  /*
   * Given a frame, load the ground truth depth map.
   */
  virtual bool loadGroundTruthDepth(const sdl::Frame& frame,
                                    cv::Mat& depth) = 0;

  void setCache(boost::filesystem::path cache_dir) { cache = cache_dir; }
  const boost::filesystem::path& getCache() { return cache; }

 protected:
  boost::filesystem::path cache;
};

class CambridgeLandmarksParser : public SceneParser {
 public:
  explicit CambridgeLandmarksParser(const boost::filesystem::path& directory)
      : directory(directory) {}
  virtual ~CambridgeLandmarksParser() {}
  void parseScene(
      std::vector<sdl::Database>& dbs,
      std::vector<std::pair<std::vector<std::reference_wrapper<sdl::Database>>,
                            std::vector<sdl::Query>>>& dbs_with_queries,
      bool inject_ground_truth) override;
  bool loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                           cv::Mat& translation) override;
  bool loadGroundTruthDepth(const sdl::Frame& frame, cv::Mat& depth) override {
    assert(false);
    return false;
  }

 private:
  boost::filesystem::path directory;
  void loadGtPosesFromFile(boost::filesystem::path pose_path, bool is_street);
  std::set<std::string> accumulateSubdirsFromFile(const std::string& filename);
  std::map<int, std::map<int, std::tuple<cv::Mat, cv::Mat>>>
      rotationsAndTranslationsByDatabaseAndFrame;
  std::map<std::string, std::reference_wrapper<Database>> dbsByName;
};

class SevenScenesParser : public SceneParser {
 public:
  explicit SevenScenesParser(const boost::filesystem::path& directory)
      : directory(directory) {}
  virtual ~SevenScenesParser() {}
  void parseScene(
      std::vector<sdl::Database>& dbs,
      std::vector<std::pair<std::vector<std::reference_wrapper<sdl::Database>>,
                            std::vector<sdl::Query>>>& dbs_with_queries,
      bool inject_ground_truth) override;
  bool loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                           cv::Mat& translation) override;
  bool loadGroundTruthDepth(const sdl::Frame& frame, cv::Mat& depth) override;

 private:
  boost::filesystem::path directory;
};

class TumParser : public SceneParser {
 public:
  explicit TumParser(const boost::filesystem::path& directory)
      : directory(directory) {}
  virtual ~TumParser() {}
  void parseScene(
      std::vector<sdl::Database>& dbs,
      std::vector<std::pair<std::vector<std::reference_wrapper<sdl::Database>>,
                            std::vector<sdl::Query>>>& dbs_with_queries,
      bool inject_ground_truth) override;

  bool loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                           cv::Mat& translation) override;
  bool loadGroundTruthDepth(const sdl::Frame& frame, cv::Mat& depth) override {
    assert(false);
    return false;
  }

 private:
  boost::filesystem::path directory;
  std::map<int, std::map<int, std::tuple<cv::Mat, cv::Mat>>>
      rotationsAndTranslationsByDatabaseAndFrame;
};

}  // namespace sdl

#endif /* SRC_DATASETS_H_ */
