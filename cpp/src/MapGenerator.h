#ifndef SRC_MAPGENERATOR_H_
#define SRC_MAPGENERATOR_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "util/DatasetReader.h"
#include "util/MinimalImage.h"
#include "util/NumType.h"

namespace sdl {

extern bool saveVariance;

typedef std::pair<dso::Vec3, float> ColoredPoint;

struct SceneCoord {
  SceneCoord()
      : coord(cv::Vec3f()),
        hasVariance(false),
        inverseDepth(-1),
        inverseVariance(-1),
        observerCam2World(dso::SE3()) {}
  explicit SceneCoord(cv::Vec3f coord)
      : coord(coord),
        hasVariance(false),
        inverseDepth(-1),
        inverseVariance(-1),
        observerCam2World(dso::SE3()) {
    assert(!saveVariance);
  }
  explicit SceneCoord(cv::Vec3f coord, float inverse_depth,
                      float inverse_variance, dso::SE3 observer_cam2world)
      : coord(coord),
        hasVariance(true),
        inverseDepth(inverse_depth),
        inverseVariance(inverse_variance),
        observerCam2World(observer_cam2world) {
    assert(saveVariance);
  }

  cv::Vec3f coord;
  bool hasVariance;
  float inverseDepth;
  float inverseVariance;
  dso::SE3 observerCam2World;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

/*
 * Sparse matrix mapping from image coordinates to scene coordinates. This is
 * functionally similar to a cv::SparseMat(2, dims, CV_32FC3), except that this
 * also stores covariance information accrued from multiple camera observations.
 */
struct SceneCoordinateMap {
  SceneCoordinateMap() : height(0), width(0) {}
  SceneCoordinateMap(int height, int width) : height(height), width(width) {}

  // TODO: it would be nice if this class wrapped the map below, but also
  // include bounds checks or something.
  int height, width;
  std::map<std::pair<int, int>, SceneCoord, std::less<std::pair<int, int>>,
           Eigen::aligned_allocator<
               std::pair<const std::pair<int, int>, SceneCoord>>>
      coords;
};
class DsoDatasetReader {
 public:
  virtual ~DsoDatasetReader() = default;
  virtual float* getPhotometricGamma() = 0;
  virtual int getNumImages() = 0;
  virtual ImageAndExposure* getImage(int id) = 0;
  virtual cv::Mat getK() = 0;
};

/*
 * Generates a map representation using DSO. This is largely copied from the
 * original DSO command line interface, and accepts similar arguments.
 */
class DsoMapGenerator {
 public:
  DsoMapGenerator(int argc, char** argv);
  explicit DsoMapGenerator(const std::string& input_path);
  DsoMapGenerator(const std::string& image_path, const std::string& calib_path,
                  bool save_non_keyframes_for_testing);
  DsoMapGenerator(cv::Mat camera_calib, int width, int height,
                  const std::vector<std::string>& image_paths,
                  const std::string& cache_path,
                  bool save_non_keyframes_for_testing);

  void runVisualOdometry(
      const std::vector<int>& indices_to_play,
      std::function<std::unique_ptr<dso::SE3>(int)> pose_loader = 0);
  void runVisualOdometry(
      std::function<std::unique_ptr<dso::SE3>(int)> pose_loader);
  void runVisualOdometry();
  inline bool hasValidPoints() const { return !pointcloud->empty(); }

  void savePointCloudAsPly(const std::string& filename);
  void savePointCloudAsPcd(const std::string& filename);
  void savePointCloudAsManyPcds(const std::string& filepath);
  void saveDepthMaps(const std::string& filepath);
  cv::Mat getCalibration() { return datasetReader->getK(); }

  void saveCameraAdjacencyList(const std::string& filename) const;
  void saveRawImages(const std::string& filepath) const;
  void savePosesInWorldFrame(const std::string& gt_filename,
                             const std::string& output_filename) const;

  int getNumImages();

  std::map<int, SceneCoordinateMap> getSceneCoordinateMaps();
  std::map<int, dso::SE3>* getPoses() { return poses.get(); }

 private:
  void parseArgument(char* arg, std::string& source, std::string& calib,
                     std::string& gamma_calib, std::string& vignette);

  int mode = 0;

  std::unique_ptr<std::list<ColoredPoint>> pointcloud;
  std::unique_ptr<std::map<
      int, std::pair<dso::SE3, std::unique_ptr<std::list<ColoredPoint>>>>>
      pointcloudsWithViewpoints;
  std::unique_ptr<std::map<int, std::unique_ptr<dso::MinimalImageF>>>
      depthImages;
  std::unique_ptr<std::map<int, std::unique_ptr<dso::MinimalImageF>>> rgbImages;
  std::unique_ptr<std::map<int, dso::SE3>> poses;
  std::unique_ptr<std::map<int, SceneCoordinateMap>> sceneCoordinateMaps;
  std::unique_ptr<std::map<int, std::set<int>>> cameraAdjacencyList;

  std::unique_ptr<DsoDatasetReader> datasetReader;
  bool saveNonKeyFramesForTesting = false;
};
}  // namespace sdl

#endif /* SRC_MAPGENERATOR_H_ */
