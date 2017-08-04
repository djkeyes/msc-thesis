
#include "CaffeDescriptor.h"

#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"

using std::string;
using std::cout;
using std::endl;
using std::runtime_error;
using std::stringstream;
using std::unique_ptr;
using std::vector;
using cv::KeyPoint;
using cv::Mat;

namespace sdl {

unique_ptr<caffe::Net<float>> readCaffeModel(
    const string& prototxt_filename, const string& caffemodel_filename) {
  caffe::NetParameter net_param;
  caffe::ReadProtoFromTextFile(prototxt_filename, &net_param);

  caffe::NetParameter weights;
  caffe::ReadProtoFromBinaryFile(caffemodel_filename, &weights);

  unique_ptr<caffe::Net<float>> net(new caffe::Net<float>(net_param));
  net->CopyTrainedLayersFrom(weights);

  net->Reshape();

  if (!net->has_blob("data")) {
    throw runtime_error("Expected blob with name data.");
  }
  auto data_layer = static_cast<caffe::MemoryDataLayer<float>*>(
      net->layer_by_name("image_data").get());

  bool has_valid_output = false;
  for (auto& output_blob : net->output_blobs()) {
    if (data_layer->width() == output_blob->width() &&
        data_layer->height() == output_blob->height()) {
      has_valid_output = true;
      break;
    }
  }

  if (!has_valid_output) {
    stringstream ss;
    ss << "Expected input dimensions of network to match output dimensions, "
          "but got input dims "
       << data_layer->blobs()[0]->shape_string()
       << " and output blobs with dims: ";
    bool first = true;
    for (auto& output_blob : net->output_blobs()) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      ss << output_blob->shape_string();
    }
    throw runtime_error(ss.str());
  }

  return net;
}

DenseDescriptorFromCaffe::DenseDescriptorFromCaffe(
    std::unique_ptr<caffe::Net<float>> net_)
    : net(std::move(net_)) {
  dataLayer = static_cast<caffe::MemoryDataLayer<float>*>(
      net->layer_by_name("image_data").get());

  for (auto& blob : net->output_blobs()) {
    if (dataLayer->width() == blob->width() &&
        dataLayer->height() == blob->height()) {
      outputBlob = blob;
      break;
    }
  }
}
int DenseDescriptorFromCaffe::descriptorSize() const {
  return outputBlob->channels();
}

void DenseDescriptorFromCaffe::detectAndCompute(cv::InputArray image,
                                                cv::InputArray mask,
                                                vector<cv::KeyPoint>& keypoints,
                                                cv::OutputArray descriptors,
                                                bool useProvidedKeypoints) {
  if (curSceneCoords.nzcount() == 0) {
    return;
  }

  if (!useProvidedKeypoints) {
    keypoints.clear();
    for (auto iter = curSceneCoords.begin(); iter != curSceneCoords.end();
         ++iter) {
      int y = iter.node()->idx[0];
      int x = iter.node()->idx[1];
      KeyPoint kpt;
      kpt.pt.x = x;
      kpt.pt.y = y;
      keypoints.push_back(kpt);
    }
  }

  if (descriptors.needed()) {
    Mat mat = image.getMat();

    std::vector<Mat> matvec;
    matvec.push_back(mat);
    vector<int> labels(1, 0);

    dataLayer->AddMatVector(matvec, labels);

    float loss;
    net->Forward(&loss);

    int channels = outputBlob->channels();
    const float* data = outputBlob->cpu_data();

    Mat& desc = descriptors.getMatRef();
    desc.create(keypoints.size(), channels, descriptorType());
    int idx = 0;
    for (const KeyPoint& kp : keypoints) {
      int row = static_cast<int>(kp.pt.y);
      int col = static_cast<int>(kp.pt.x);

      for (int c = 0; c < channels; ++c) {
        int offset = outputBlob->offset(0, c, row, col);
        desc.at<float>(idx, c) = data[offset];
      }
      ++idx;
    }
  }
}
}  // namespace sdl
