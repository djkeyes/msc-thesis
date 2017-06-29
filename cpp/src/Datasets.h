
#ifndef SRC_DATASETS_H_
#define SRC_DATASETS_H_

#include <map>
#include <tuple>
#include <tuple>
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

enum MappingMethod { DSO, NONE };

class SceneParser {
 public:
  explicit SceneParser(MappingMethod mapping_method)
      : mappingMethod(mapping_method) {}
  virtual ~SceneParser() {}

  /*
   * Parse a scene into a set of databases and a set of queries (which may be
   * built from overlapping data).
   */
  virtual void parseScene(std::vector<sdl::Database>& dbs,
                          std::vector<sdl::Query>& queries) = 0;

  /*
   * Given a frame, load the ground truth pose (as a rotation and translation
   * matrix) from the dataset.
   */
  virtual void loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                                   cv::Mat& translation) = 0;

 protected:
  MappingMethod mappingMethod;
};

class SevenScenesParser : public SceneParser {
 public:
  explicit SevenScenesParser(const boost::filesystem::path& directory,
                             MappingMethod mapping_method)
      : SceneParser(mapping_method), directory(directory) {}
  virtual ~SevenScenesParser() {}
  virtual void parseScene(std::vector<sdl::Database>& dbs,
                          std::vector<sdl::Query>& queries);
  virtual void loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                                   cv::Mat& translation);
  void setCache(boost::filesystem::path cache_dir) { cache = cache_dir; }

 private:
  boost::filesystem::path directory;
  boost::filesystem::path cache;
};

class TumParser : public SceneParser {
 public:
  explicit TumParser(const boost::filesystem::path& directory,
                     MappingMethod mapping_method)
      : SceneParser(mapping_method), directory(directory) {}
  virtual ~TumParser() {}
  virtual void parseScene(std::vector<sdl::Database>& dbs,
                          std::vector<sdl::Query>& queries);

  virtual void loadGroundTruthPose(const sdl::Frame& frame, cv::Mat& rotation,
                                   cv::Mat& translation);

  void setCache(boost::filesystem::path cache_dir) { cache = cache_dir; }

 private:
  boost::filesystem::path directory;
  boost::filesystem::path cache;
  std::map<int, std::map<int, std::tuple<cv::Mat, cv::Mat>>>
      rotationsAndTranslationsByDatabaseAndFrame;
};

}  // namespace sdl

#endif /* SRC_DATASETS_H_ */
