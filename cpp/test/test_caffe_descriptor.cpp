
#include "CaffeDescriptor.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/mat.hpp"

using std::string;
using std::cout;
using std::endl;
using std::unique_ptr;
using std::make_pair;
using std::move;
using std::vector;
using cv::Mat;
using cv::KeyPoint;
using cv::noArray;
using cv::SparseMat;

bool test_loading_simple_caffenet() {
  unique_ptr<caffe::Net<float>> net = sdl::readCaffeModel(
      "/home/daniel/git/msc-thesis/python/DescriptorLearning/"
      "test.prototxt",
      "/home/daniel/experiments/denseCorrespondence/"
      "snap_adam_iter_23000.caffemodel");

  sdl::DenseDescriptorFromCaffe descriptor_extractor(move(net));

  // These have to conform exactly to the prototxt. Not sure how to resize them.
  int height = 480;
  int width = 640;
  Mat img = Mat::zeros(height, width, CV_8UC3);

  sdl::SceneCoordinateMap scene_coords(height, width);
  scene_coords.coords[make_pair(0, 0)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  scene_coords.coords[make_pair(5, 5)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  scene_coords.coords[make_pair(20, 0)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  scene_coords.coords[make_pair(0, 60)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  descriptor_extractor.setCurrentSceneCoords(scene_coords);

  vector<KeyPoint> keypoints;
  Mat descriptors;
  descriptor_extractor.detectAndCompute(img, noArray(), keypoints, descriptors);

  assert(keypoints.size() == 4);
  assert(keypoints.size() == scene_coords.coords.size());
  assert(static_cast<int>(keypoints.size()) == descriptors.rows);
  unsigned int num_matched_keypoints = 0;
  for (const auto& keypoint : keypoints) {
    for (const auto& element : scene_coords.coords) {
      int row = element.first.first;
      int col = element.first.second;
      if (row == keypoint.pt.y && col == keypoint.pt.x) {
        num_matched_keypoints++;
        break;
      }
    }
  }
  assert(num_matched_keypoints == keypoints.size());

  return false;
}

void save_1layer_identity_cnn(const string& prototxt_filename,
                              const string& caffemodel_filename) {
  caffe::NetParameter net_param;
  net_param.set_name("dummy-net");

  caffe::MemoryDataParameter* mem_data(new caffe::MemoryDataParameter());
  mem_data->set_batch_size(1);
  mem_data->set_channels(3);
  mem_data->set_height(480);
  mem_data->set_width(640);
  caffe::LayerParameter* mem_data_layer(net_param.add_layer());
  mem_data_layer->set_name("image_data");
  mem_data_layer->set_type("MemoryData");
  mem_data_layer->add_top("data");
  mem_data_layer->add_top("label_ignore");
  mem_data_layer->set_allocated_memory_data_param(mem_data);

  caffe::ConvolutionParameter* conv = new caffe::ConvolutionParameter();
  conv->set_num_output(3);
  conv->add_kernel_size(1);
  conv->add_stride(1);
  caffe::LayerParameter* conv_layer(net_param.add_layer());
  conv_layer->set_name("conv");
  conv_layer->set_type("Convolution");
  conv_layer->add_bottom("data");
  conv_layer->add_top("upsample_1");
  conv_layer->set_allocated_convolution_param(conv);

  caffe::WriteProtoToTextFile(net_param, prototxt_filename);

  caffe::Net<float> net(net_param);
  auto& lay = net.layer_by_name("conv");
  cout << "conv blob size: " << lay->blobs().size() << endl;
  cout << "first blob shape: " << lay->blobs()[0]->shape_string() << endl;
  cout << "second blob shape: " << lay->blobs()[1]->shape_string() << endl;
  // Pretty sure blobs()[0] are the conv weights, and blobs()[1] are the biases.
  float* data = lay->blobs()[0]->mutable_cpu_data();
  // just fill with identity
  for (int filter_idx = 0; filter_idx < lay->blobs()[0]->shape(0);
       filter_idx++) {
    for (int input_channel = 0; input_channel < lay->blobs()[0]->shape(1);
         input_channel++) {
      if (filter_idx == input_channel) {
        data[lay->blobs()[0]->offset(filter_idx, input_channel)] = 1;
      } else {
        data[lay->blobs()[0]->offset(filter_idx, input_channel)] = 0;
      }
    }
  }
  caffe::NetParameter net_param_with_weights;
  net.ToProto(&net_param_with_weights);
  caffe::WriteProtoToBinaryFile(net_param_with_weights, caffemodel_filename);
}
bool test_single_layer_net() {
  // TODO: pass weights as a matrix?
  save_1layer_identity_cnn("dummy.prototxt", "dummy.caffemodel");
  unique_ptr<caffe::Net<float>> net =
      sdl::readCaffeModel("dummy.prototxt", "dummy.caffemodel");

  sdl::DenseDescriptorFromCaffe descriptor_extractor(move(net));

  // These have to conform exactly to the prototxt. Not sure how to resize them.
  int dims[2] = {480, 640};
  Mat img = Mat::zeros(dims[0], dims[1], CV_8UC3);
  img.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 3);
  img.at<cv::Vec3b>(5, 5) = cv::Vec3b(4, 5, 6);
  img.at<cv::Vec3b>(20, 0) = cv::Vec3b(7, 8, 9);
  img.at<cv::Vec3b>(0, 60) = cv::Vec3b(253, 254, 255);

  sdl::SceneCoordinateMap scene_coords(dims[0], dims[1]);
  scene_coords.coords[make_pair(0, 0)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  scene_coords.coords[make_pair(5, 5)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  scene_coords.coords[make_pair(20, 0)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  scene_coords.coords[make_pair(0, 60)] = sdl::SceneCoord(cv::Vec3f(1, 2, 3));
  descriptor_extractor.setCurrentSceneCoords(scene_coords);

  vector<KeyPoint> keypoints;
  Mat descriptors;
  descriptor_extractor.detectAndCompute(img, noArray(), keypoints, descriptors);

  assert(keypoints.size() == 4);
  assert(keypoints.size() == scene_coords.coords.size());
  assert(static_cast<int>(keypoints.size()) == descriptors.rows);
  unsigned int num_matched_keypoints = 0;
  for (const auto& keypoint : keypoints) {
    for (const auto& element : scene_coords.coords) {
      int row = element.first.first;
      int col = element.first.second;
      if (row == keypoint.pt.y && col == keypoint.pt.x) {
        num_matched_keypoints++;
        break;
      }
    }
  }
  assert(num_matched_keypoints == keypoints.size());
  assert(descriptors.cols == 3);
  // this should be the identity
  for (unsigned int i = 0; i < keypoints.size(); ++i) {
    const auto& pt = keypoints[i].pt;
    if (pt.y == 0 && pt.x == 0) {
      assert(descriptors.at<float>(i, 0) == 1);
      assert(descriptors.at<float>(i, 1) == 2);
      assert(descriptors.at<float>(i, 2) == 3);
    } else if (pt.y == 5 && pt.x == 5) {
      assert(descriptors.at<float>(i, 0) == 4);
      assert(descriptors.at<float>(i, 1) == 5);
      assert(descriptors.at<float>(i, 2) == 6);
    } else if (pt.y == 20 && pt.x == 0) {
      assert(descriptors.at<float>(i, 0) == 7);
      assert(descriptors.at<float>(i, 1) == 8);
      assert(descriptors.at<float>(i, 2) == 9);
    } else if (pt.y == 0 && pt.x == 60) {
      assert(descriptors.at<float>(i, 0) == 253);
      assert(descriptors.at<float>(i, 1) == 254);
      assert(descriptors.at<float>(i, 2) == 255);
    } else {
      assert(false);
    }
  }

  return false;
}

int main(int argc, char** argv) {
  bool any_tests_failed = false;
  sdl::saveVariance = false;
  any_tests_failed |= test_loading_simple_caffenet();
  any_tests_failed |= test_single_layer_net();
  return any_tests_failed;
}
