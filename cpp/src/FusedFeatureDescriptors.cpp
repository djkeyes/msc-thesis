#include <algorithm>
#include <list>

#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"

#include "FusedFeatureDescriptors.h"

using namespace cv;
using std::vector;
using std::list;

namespace sdl {
NearestDescriptorAssigner::NearestDescriptorAssigner(Feature2D& feature_) :
		feature(feature_) {
}

void NearestDescriptorAssigner::detectAndCompute(InputArray image_, InputArray mask, vector<KeyPoint>& keypoints,
		OutputArray descriptors, bool useProvidedKeypoints) {
	if (curSceneCoords.nzcount() == 0) {
		return;
	}

	if (!useProvidedKeypoints) {

		Mat image = image_.getMat();
		vector<KeyPoint> initial_keypoints;
		feature.detect(image, initial_keypoints);

		double dist_threshold = 3;
		double dist_threshold_sq = dist_threshold * dist_threshold;
		// We want to do a 2D threshold nearest neighbor search for each depth
		// value. Ideally, we would like something that works for both sparse
		// keypoints (eg sift) and dense keypoints. Some options:
		// -linear search through all detected keypoints (very slow for dense)
		// -linear search through all adjacent pixels (slow for sparse)
		// -bucket descriptors and linear search through adjacent buckets (slow for dense)
		// -kd tree (very slow for dense)
		// For now, we'll do the bucketing approach.

		// each bucket is size dist_threshold x dist_threshold
		int bucket_rows = floor(image.size().height / dist_threshold) + 1;
		int bucket_cols = floor(image.size().width / dist_threshold) + 1;
		vector<vector<list<const KeyPoint*>>> buckets(bucket_rows, vector<list<const KeyPoint*>>(bucket_cols, list<const KeyPoint*>()));

		for (const auto& keypoint : initial_keypoints) {
			int row = floor(keypoint.pt.y / dist_threshold);
			int col = floor(keypoint.pt.x / dist_threshold);

			buckets[row][col].push_back(&keypoint);
		}

		keypoints.clear();
		for (auto iter = curSceneCoords.begin(); iter != curSceneCoords.end(); ++iter) {
			int y = iter.node()->idx[0];
			int x = iter.node()->idx[1];

			int row = floor(y / dist_threshold);
			int col = floor(x / dist_threshold);

			const KeyPoint* closest;
			double closest_distsq = dist_threshold_sq;
			// only need to check from row-1 to row+1 and col-1 to col+1
			for (int bucket_row = std::max(0, row - 1); bucket_row <= std::min(bucket_rows - 1, row + 1);
					++bucket_row) {
				for (int bucket_col = std::max(0, col - 1); bucket_col <= std::min(bucket_cols - 1, col + 1);
						++bucket_col) {
					for (const auto& keypoint : buckets[bucket_row][bucket_col]) {
						double dx = keypoint->pt.x - x;
						double dy = keypoint->pt.y - y;
						double distsq = dx * dx + dy * dy;
						if (distsq < dist_threshold_sq && distsq < closest_distsq) {
							closest = keypoint;
							closest_distsq = distsq;
						}
					}
				}
			}

			if (closest_distsq < dist_threshold_sq) {
				KeyPoint kpt(*closest);
				kpt.pt.x = x;
				kpt.pt.y = y;
				keypoints.push_back(kpt);
			}
		}

	}

	if (descriptors.needed()) {
		// This computes the descriptors at the slightly shifted locations
		feature.compute(image_, keypoints, descriptors);
	}

}

}

