#ifndef SRC_CUSTOMSIFTKEYPOINTS_H_
#define SRC_CUSTOMSIFTKEYPOINTS_H_

#include <vector>

namespace sdl {
/*
 * Given a set of keypoints containing only pixel locations, finds extrema
 * in the scale space, and computes dominant gradient direction, so that
 * the resultant SIFT descriptors are actually scale and rotation invariant.
 * This is largely copied from the nonfree opencv_contrib code, since that
 * implementation isn't very easily extensible.
 */
void computeSiftOrientationAndScale(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, int num_octave_layers = 3,
		double sigma = 1.6);

}

#endif /* SRC_CUSTOMSIFTKEYPOINTS_H_ */
