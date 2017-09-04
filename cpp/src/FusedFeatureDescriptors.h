#ifndef SRC_FUSEDFEATUREDESCRIPTORS_H_
#define SRC_FUSEDFEATUREDESCRIPTORS_H_

#include <vector>

#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"

#include "MapGenerator.h"

namespace sdl {
/*
 * Feature detector that allows users to specify an associated map of per-pixel
 * world coordinates before the descriptor computation step.
 */
class SceneCoordFeatureDetector : public cv::Feature2D {
 public:
  virtual ~SceneCoordFeatureDetector() = default;

  void setCurrentSceneCoords(SceneCoordinateMap cur_scene_coords) {
    curSceneCoords = cur_scene_coords;
  }

 protected:
  SceneCoordinateMap curSceneCoords;
};

/*
 * Given a Feature2D, creates new keypoints at defined locations in the depth
 * map, assigned to the nearest descriptor (within some threshold).
 */
class NearestDescriptorAssigner : public SceneCoordFeatureDetector {
 public:
  explicit NearestDescriptorAssigner(cv::Feature2D& feature);
  ~NearestDescriptorAssigner() = default;

  int descriptorSize() const override { return feature.descriptorSize(); }

  int descriptorType() const override { return feature.descriptorType(); }

  int defaultNorm() const override { return feature.defaultNorm(); }

  void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false) override;

 private:
  cv::Feature2D& feature;
};

}  // namespace sdl

#endif /* SRC_FUSEDFEATUREDESCRIPTORS_H_ */
