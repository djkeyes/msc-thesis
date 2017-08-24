
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Datasets.h"
#include "MapGenerator.h"
#include "Relocalization.h"
#include "boost/filesystem.hpp"

using namespace cv;
using namespace std;

/*
 * Functional test to check that valid scene coordinates are produced.
 */

void test_scene_coordinates_have_valid_projection(
    const sdl::SceneCoordinateMap& scene_coords, const dso::SE3& pose, cv::Mat K) {
  assert(scene_coords.coords.size() > 0);
  for (const auto& element : scene_coords.coords) {
    int row = element.first.first;
    int col = element.first.second;

    cv::Vec3f point_cv = element.second;
    dso::Vec3 world_point(point_cv[0], point_cv[1], point_cv[2]);

    // we could use pose.invere() * world_point directly, but this demonstrates
    // operations using a 3x4 transform matrix
    Eigen::Matrix<double, 3, 4> transform = pose.matrix3x4();
    Eigen::Matrix3d R = transform.block(0, 0, 3, 3);
    Eigen::Vector3d t = transform.col(3);
    // invert
    Eigen::Matrix3d Rinv = R.transpose();
    Eigen::Vector3d tinv = -Rinv * t;

    dso::Vec3 local_point = Rinv * world_point + tinv;

    assert(local_point[2] > 0);

    double x =
        local_point[0] * K.at<float>(0, 0) / local_point[2] + K.at<float>(0, 2);
    double y =
        local_point[1] * K.at<float>(1, 1) / local_point[2] + K.at<float>(1, 2);

    double dx = fabs(x - col);
    double dy = fabs(y - row);

    if (dx > 1 || dy > 1) {
      cout << "(" << col << ", " << row << ") --> (" << world_point[0] << ", "
           << world_point[1] << ", " << world_point[2] << ") --> ("
           << local_point[0] << ", " << local_point[1] << ", " << local_point[2]
           << ") --> (" << x << ", " << y << ")" << endl;
    }
    assert(dx <= 1);
    assert(dy <= 1);
  }
}

class FrameCleanup {
 public:
  explicit FrameCleanup(const sdl::Frame& f) : f_(f) {}
  ~FrameCleanup() {
    boost::filesystem::remove(f_.getSceneCoordinateFilename());
  }

 private:
  const sdl::Frame& f_;
};
void test_load_scene_coordinates_from_save(const sdl::Frame& f,
                                           const sdl::SceneCoordinateMap& scene_coords) {
  FrameCleanup fc(f);

  f.saveSceneCoordinates(scene_coords);
//  sdl::SceneCoordinateMap loaded = f.loadSceneCoordinates();
//
  // need to change signature of loadSceneCoordinates
  assert(false);
//  assert(loaded.dims() == scene_coords.dims());
//  assert(loaded.dims() == 2);
//  for (int i = 0; i < loaded.dims(); ++i) {
//    assert(loaded.size(i) == scene_coords.size(i));
//  }
//
//  for (int row = 0; row < loaded.size(0); ++row) {
//    for (int col = 0; col < loaded.size(1); ++col) {
//      for (int channel = 0; channel < 3; ++channel) {
//        assert(loaded.value<cv::Vec3f>(row, col).val[channel] ==
//               scene_coords.value<cv::Vec3f>(row, col).val[channel]);
//      }
//    }
//  }
}

bool test_map_generation(const string& data_dir, const string& cache_dir) {
  unique_ptr<sdl::TumParser> parser(
      new sdl::TumParser(data_dir));
  parser->setCache(cache_dir);

  vector<sdl::Database> dbs;
  vector<pair<vector<reference_wrapper<sdl::Database>>, vector<sdl::Query>>>
      dbs_with_queries;

  parser->parseScene(dbs, dbs_with_queries, false);
  for (int i = dbs.size() - 1; i >= 0; --i) {
    auto& db = dbs[i];

    sdl::DsoMapGenerator* map_gen =
        static_cast<sdl::DsoMapGenerator*>(db.getMapper());
    map<int, unique_ptr<sdl::Frame>>* frames = db.getFrames();

    // first run the full mapping algorithm
    map_gen->runVisualOdometry();

    map<int, sdl::SceneCoordinateMap> scene_coordinate_maps =
        map_gen->getSceneCoordinateMaps();
    map<int, dso::SE3>* poses = map_gen->getPoses();

    for (unsigned int frame_id = 0; frame_id < frames->size(); ++frame_id) {
      auto iter = scene_coordinate_maps.find(frame_id);
      if (iter == scene_coordinate_maps.end() ||
          poses->find(frame_id) == poses->end()) {
        continue;
      }

      sdl::Frame* f = frames->at(frame_id).get();

      cv::Mat K = map_gen->getCalibration();
      test_load_scene_coordinates_from_save(*f, iter->second);
      test_scene_coordinates_have_valid_projection(iter->second,
                                                   poses->at(frame_id), K);
    }

    // Could also check that adjacent frames have points that project into each
    // others' coordinate frames.

    // pop from the vector, so that old results are erased
    dbs.pop_back();
  }
  return false;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "usage: test_map_generation [data_dir] [cache_dir]" << endl;
    return 1;
  }

  string data_dir(argv[1]);
  string cache_dir(argv[2]);

  bool any_tests_failed = false;
  any_tests_failed |= test_map_generation(data_dir, cache_dir);
  return any_tests_failed;
}
