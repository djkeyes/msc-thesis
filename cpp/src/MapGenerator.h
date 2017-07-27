#ifndef SRC_MAPGENERATOR_H_
#define SRC_MAPGENERATOR_H_

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

typedef std::pair<dso::Vec3, float> ColoredPoint;

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
  DsoMapGenerator(cv::Mat camera_calib, int width, int height,
                  const std::vector<std::string>& image_paths,
                  const std::string& cache_path);

  void runVisualOdometry(const std::vector<int>& indices_to_play);
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

  std::map<int, cv::SparseMat> getSceneCoordinateMaps();
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
  std::unique_ptr<std::map<int, cv::SparseMat>> sceneCoordinateMaps;
  std::unique_ptr<std::map<int, std::set<int>>> cameraAdjacencyList;

  std::unique_ptr<DsoDatasetReader> datasetReader;
};
}  // namespace sdl

#endif /* SRC_MAPGENERATOR_H_ */
