#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"

#include "FusedFeatureDescriptors.h"

using namespace cv;
using std::vector;

namespace sdl {
NearestDescriptorAssigner::NearestDescriptorAssigner(Feature2D feature_) :
		feature(feature_) {
}

void NearestDescriptorAssigner::detectAndCompute(InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
		OutputArray descriptors, bool useProvidedKeypoints) {

//	feature->detect(image, keypoints);
//	for (auto iter = scene_coords.begin(); iter != scene_coords.end(); ++iter) {
//		int y = iter.node()->idx[0];
//		int x = iter.node()->idx[1];
//		keypoints.push_back(KeyPoint(x, y, 1));
//	}

}

}

