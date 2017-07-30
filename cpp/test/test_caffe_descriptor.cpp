
#include "CaffeDescriptor.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "opencv2/core/mat.hpp"

using std::cout;
using std::endl;
using std::unique_ptr;
using std::move;
using std::vector;
using cv::Mat;
using cv::KeyPoint;
using cv::noArray;
using cv::SparseMat;
using cv::Vec3f;

bool test_loading_simple_caffenet() {
  // TODO(daniel): write a simple caffemodel and load that.
  unique_ptr<caffe::Net<float>> net = sdl::readCaffeModel(
      "/home/daniel/git/msc-thesis/python/DescriptorLearning/"
      "test.prototxt",
      "/home/daniel/experiments/denseCorrespondence/"
      "snap_fullres_labscene_iter_1000.caffemodel");

  sdl::DenseDescriptorFromCaffe descriptor_extractor(move(net));

  // These have to conform exactly to the prototxt. Not sure how to resize them.
  int dims[2] = {480, 640};
  Mat img = Mat::zeros(dims[0], dims[1], CV_8UC3);

  SparseMat scene_coords(2, dims, CV_32FC3);
  scene_coords.ref<Vec3f>(0, 0) = Vec3f(1, 2, 3);
  scene_coords.ref<Vec3f>(5, 5) = Vec3f(1, 2, 3);
  scene_coords.ref<Vec3f>(20, 0) = Vec3f(1, 2, 3);
  scene_coords.ref<Vec3f>(0, 60) = Vec3f(1, 2, 3);
  descriptor_extractor.setCurrentSceneCoords(scene_coords);

  vector<KeyPoint> keypoints;
  Mat descriptors;
  descriptor_extractor.detectAndCompute(img, noArray(), keypoints, descriptors);

  assert(keypoints.size() == 4);
  assert(keypoints.size() == scene_coords.nzcount());
  assert(static_cast<int>(keypoints.size()) == descriptors.rows);
  int num_matched_keypoints = 0;
  for (const auto& keypoint : keypoints) {
    for (auto iter = scene_coords.begin(); iter != scene_coords.end(); ++iter) {
      int* index = iter.node()->idx;
      if (index[0] == keypoint.pt.y && index[1] == keypoint.pt.x) {
        num_matched_keypoints++;
        break;
      }
    }
  }
  assert(num_matched_keypoints == keypoints.size());

  return false;
}

int main(int argc, char** argv) {
  bool any_tests_failed = false;
  any_tests_failed |= test_loading_simple_caffenet();
  return any_tests_failed;
}
