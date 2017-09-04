
#ifndef SRC_CAFFEDESCRIPTOR_H_
#define SRC_CAFFEDESCRIPTOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"

#include "FusedFeatureDescriptors.h"

namespace sdl {

std::unique_ptr<caffe::Net<float>> readCaffeModel(
    const std::string& prototxt_filename,
    const std::string& caffemodel_filename);

/*
 * Descriptor backed by a trained caffe FCN model.
 *
 * This assumes the underlying caffe model produces a dense descriptor.
 * Consequentially, the user can specify a few keypoints at which to
 * evaluate
 * the descriptor, and this simply returns that subset of pixels.
 */
class DenseDescriptorFromCaffe : public SceneCoordFeatureDetector {
 public:
  explicit DenseDescriptorFromCaffe(std::unique_ptr<caffe::Net<float>> net_);
  ~DenseDescriptorFromCaffe() = default;

  int descriptorSize() const override;

  int descriptorType() const override { return CV_32F; };

  void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false) override;

  // releases the net to free up memory. Subsequent descriptor evaluations will
  // fail.
  void freeNet() { net.reset(); }

 private:
  int outputSize;
  std::unique_ptr<caffe::Net<float>> net;
  caffe::MemoryDataLayer<float>* dataLayer;
  caffe::Blob<float>* outputBlob;
  std::vector<float> data;
};

}  // namespace sdl

#endif /* SRC_CAFFEDESCRIPTOR_H_ */
