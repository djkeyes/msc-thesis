#ifndef SRC_FUSEDFEATUREDESCRIPTORS_H_
#define SRC_FUSEDFEATUREDESCRIPTORS_H_

#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"

namespace sdl {
/*
 * Feature detector that allows users to specify an associated map of per-pixel
 * world coordinates before the descriptor computation step.
 */
class SceneCoordFeatureDetector: public cv::Feature2D {
public:
	virtual ~SceneCoordFeatureDetector() = default;

	void setCurrentSceneCoords(cv::SparseMat cur_scene_coords) {
		curSceneCoords = cur_scene_coords;
	}

protected:
	cv::SparseMat curSceneCoords;
};

/*
 * Given a Feature2D, creates new keypoints at defined locations in the depth
 * map, assigned to the nearest descriptor (within some threshold).
 */
class NearestDescriptorAssigner: public SceneCoordFeatureDetector {

public:
	NearestDescriptorAssigner(cv::Feature2D feature);
	~NearestDescriptorAssigner() = default;

	void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
			cv::OutputArray descriptors, bool useProvidedKeypoints = false) override;
private:
	cv::Feature2D& feature;
};

}
#endif /* SRC_FUSEDFEATUREDESCRIPTORS_H_ */
