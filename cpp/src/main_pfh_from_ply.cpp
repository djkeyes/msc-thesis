#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>

using namespace pcl;
using namespace std;

/*
 * Extend NormalEstimation to provide some statistics about the computation time.
 */
template<typename PointInT, typename PointOutT>
class NormalEstimationWithDebugInfo: public NormalEstimation<PointInT, PointOutT> {

	using Feature<PointInT, PointOutT>::indices_;
	using Feature<PointInT, PointOutT>::input_;
	using Feature<PointInT, PointOutT>::surface_;
	using Feature<PointInT, PointOutT>::k_;
	using Feature<PointInT, PointOutT>::search_parameter_;
	using NormalEstimation<PointInT, PointOutT>::vpx_;
	using NormalEstimation<PointInT, PointOutT>::vpy_;
	using NormalEstimation<PointInT, PointOutT>::vpz_;

	typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;

	void computeFeature(PointCloudOut &output) override
	{
		vector<int> nn_indices(k_);
		vector<float> nn_dists(k_);

		output.is_dense = true;
		// Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
		if (input_->is_dense) {
			// Iterating over the entire index vector
			for (size_t idx = 0; idx < indices_->size(); ++idx) {
				if (this->searchForNeighbors((*indices_)[idx],
						search_parameter_, nn_indices, nn_dists) == 0) {
					output.points[idx].normal[0] =
							output.points[idx].normal[1] =
									output.points[idx].normal[2] =
											output.points[idx].curvature =
													numeric_limits<float>::quiet_NaN();

					output.is_dense = false;
					printProgress(idx, nn_indices.size());
					continue;
				}

				this->computePointNormal(*surface_, nn_indices,
						output.points[idx].normal[0],
						output.points[idx].normal[1],
						output.points[idx].normal[2],
						output.points[idx].curvature);

				flipNormalTowardsViewpoint(input_->points[(*indices_)[idx]],
						vpx_, vpy_, vpz_, output.points[idx].normal[0],
						output.points[idx].normal[1],
						output.points[idx].normal[2]);

				printProgress(idx, nn_indices.size());
			}
		} else {
			// Iterating over the entire index vector
			for (size_t idx = 0; idx < indices_->size(); ++idx) {
				if (!isFinite((*input_)[(*indices_)[idx]])
						|| this->searchForNeighbors((*indices_)[idx],
								search_parameter_, nn_indices, nn_dists) == 0) {
					output.points[idx].normal[0] =
							output.points[idx].normal[1] =
									output.points[idx].normal[2] =
											output.points[idx].curvature =
													numeric_limits<float>::quiet_NaN();

					output.is_dense = false;
					printProgress(idx, nn_indices.size());
					continue;
				}

				this->computePointNormal(*surface_, nn_indices,
						output.points[idx].normal[0],
						output.points[idx].normal[1],
						output.points[idx].normal[2],
						output.points[idx].curvature);

				flipNormalTowardsViewpoint(input_->points[(*indices_)[idx]],
						vpx_, vpy_, vpz_, output.points[idx].normal[0],
						output.points[idx].normal[1],
						output.points[idx].normal[2]);

				printProgress(idx, nn_indices.size());
			}
		}
	}

private:
	long long totalNNs = 0;
	long long fewestNNs = -1;
	long long mostNNs = -1;
	void printProgress(int current_idx, int num_nns) {
		totalNNs += num_nns;
		if (current_idx == 0) {
			fewestNNs = mostNNs = static_cast<long long>(num_nns);
		} else {
			fewestNNs = std::min(fewestNNs, static_cast<long long>(num_nns));
			mostNNs = std::max(mostNNs, static_cast<long long>(num_nns));
		}
		int total = indices_->size();
		if (current_idx % 1000 == 0 || current_idx == total - 1) {
			cout << static_cast<int>(100 * current_idx / total) << "% ("
					<< current_idx << "/" << total << "), found " << num_nns
					<< " nearest neighbors. Avg #nns = "
					<< static_cast<double>(totalNNs) / (current_idx + 1)
					<< ", min #nns = " << fewestNNs << ", max #nns = "
					<< mostNNs << endl;
		}
	}
};

template<typename PointInT, typename PointNT,
		typename PointOutT = pcl::PFHSignature125>
class PFHEstimationWithDebugInfo: public PFHEstimation<PointInT, PointNT,
		PointOutT> {

	using Feature<PointInT, PointOutT>::indices_;
	using Feature<PointInT, PointOutT>::input_;
	using Feature<PointInT, PointOutT>::surface_;
	using Feature<PointInT, PointOutT>::k_;
	using Feature<PointInT, PointOutT>::search_parameter_;
	using PFHEstimation<PointInT, PointNT, PointOutT>::feature_map_;
	using PFHEstimation<PointInT, PointNT, PointOutT>::key_list_;
	using PFHEstimation<PointInT, PointNT, PointOutT>::normals_;
	using PFHEstimation<PointInT, PointNT, PointOutT>::nr_subdiv_;
	using PFHEstimation<PointInT, PointNT, PointOutT>::pfh_histogram_;

	typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;

	void computeFeature(PointCloudOut &output) override
	{
		// Clear the feature map
		feature_map_.clear();
		queue<pair<int, int> > empty;
		swap(key_list_, empty);

		pfh_histogram_.setZero(nr_subdiv_ * nr_subdiv_ * nr_subdiv_);

		// Allocate enough space to hold the results
		// \note This resize is irrelevant for a radiusSearch ().
		vector<int> nn_indices(k_);
		vector<float> nn_dists(k_);

		output.is_dense = true;
		// Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
		if (input_->is_dense) {
			// Iterating over the entire index vector
			for (size_t idx = 0; idx < indices_->size(); ++idx) {
				if (this->searchForNeighbors((*indices_)[idx],
						search_parameter_, nn_indices, nn_dists) == 0) {
					for (int d = 0; d < pfh_histogram_.size(); ++d)
						output.points[idx].histogram[d] =
								numeric_limits<float>::quiet_NaN();

					output.is_dense = false;
					printProgress(idx, nn_indices.size());
					continue;
				}

				// Estimate the PFH signature at each patch
				this->computePointPFHSignature(*surface_, *normals_, nn_indices,
						nr_subdiv_, pfh_histogram_);

				// Copy into the resultant cloud
				for (int d = 0; d < pfh_histogram_.size(); ++d) {
					output.points[idx].histogram[d] = pfh_histogram_[d];
				}
				printProgress(idx, nn_indices.size());
			}
		} else {
			// Iterating over the entire index vector
			for (size_t idx = 0; idx < indices_->size(); ++idx) {
				if (!isFinite((*input_)[(*indices_)[idx]])
						|| !isFinite((*normals_)[(*indices_)[idx]])
						|| this->searchForNeighbors((*indices_)[idx],
								search_parameter_, nn_indices, nn_dists) == 0) {
					for (int d = 0; d < pfh_histogram_.size(); ++d)
						output.points[idx].histogram[d] =
								numeric_limits<float>::quiet_NaN();

					output.is_dense = false;
					printProgress(idx, nn_indices.size());
					continue;
				}

				// Estimate the PFH signature at each patch
				this->computePointPFHSignature(*surface_, *normals_, nn_indices,
						nr_subdiv_, pfh_histogram_);

				// Copy into the resultant cloud
				for (int d = 0; d < pfh_histogram_.size(); ++d) {
					output.points[idx].histogram[d] = pfh_histogram_[d];
				}
				printProgress(idx, nn_indices.size());
			}
		}
	}

private:
	long long totalNNs = 0;
	long long fewestNNs = -1;
	long long mostNNs = -1;
	void printProgress(int current_idx, int num_nns) {
		totalNNs += num_nns;
		if (current_idx == 0) {
			fewestNNs = mostNNs = static_cast<long long>(num_nns);
		} else {
			fewestNNs = std::min(fewestNNs, static_cast<long long>(num_nns));
			mostNNs = std::max(mostNNs, static_cast<long long>(num_nns));
		}
		int total = indices_->size();
		if (current_idx % 1000 == 0 || current_idx == total - 1) {
			cout << static_cast<int>(100 * current_idx / total) << "% ("
					<< current_idx << "/" << total << "), found " << num_nns
					<< " nearest neighbors. Avg #nns = "
					<< static_cast<double>(totalNNs) / (current_idx + 1)
					<< ", min #nns = " << fewestNNs << ", max #nns = "
					<< mostNNs << endl;
		}
	}
};

PointCloud<Normal>::Ptr estimateNormals(PointCloud<PointXYZI>::ConstPtr cloud,
		double search_radius) {
	// TODO: can use NormalEstimateOMP for drop-in multicore speedup
	NormalEstimationWithDebugInfo<PointXYZI, Normal> ne;
	ne.setInputCloud(cloud);

	search::KdTree<PointXYZI>::Ptr tree(new search::KdTree<PointXYZI>());
	ne.setSearchMethod(tree);

	ne.setRadiusSearch(search_radius);

	PointCloud<Normal>::Ptr cloud_normals(new PointCloud<Normal>);

	// TODO: record viewpoint of each point, so we can compute the sign of the
	// normal correctly with setViewPoint() or similar

	ne.compute(*cloud_normals);

	return cloud_normals;
}

PointCloud<PFHSignature125>::Ptr estimatePfh(PointCloud<PointXYZI>::Ptr cloud,
		PointCloud<Normal>::Ptr cloud_normals, double search_radius) {
	PFHEstimationWithDebugInfo<PointXYZI, Normal, PFHSignature125> pfh;
	pfh.setInputCloud(cloud);
	pfh.setInputNormals(cloud_normals);

	// Create an empty kdtree representation, and pass it to the PFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	search::KdTree<PointXYZI>::Ptr tree(new search::KdTree<PointXYZI>());
	pfh.setSearchMethod(tree);

	// Output datasets
	PointCloud<PFHSignature125>::Ptr pfhs(new PointCloud<PFHSignature125>());

	// Use all neighbors in a sphere of radius 5cm
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	pfh.setRadiusSearch(search_radius);

	// Compute the features
	pfh.compute(*pfhs);

	return pfhs;
}

int main(int argc, char** argv) {
	PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);

	string filename = "map-no-pattern.ply";
	if (io::loadPLYFile<PointXYZI>(filename, *cloud) == -1) //* load the file
			{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return (-1);
	}
	cout << "Loaded " << cloud->width * cloud->height << " data points from "
			<< filename << endl;

	// the flann kd search tree is somehow built incrementally based on the
	// pointcloud index ordering. randomize the order, otherwise the tree will
	// be very slow for points far from the initial camera position
	random_shuffle(cloud->begin(), cloud->end());

	// note: the normal computation search radius must be smaller than the PFH search radius
	double normalRadius = 0.05;
	double pfhRadius = normalRadius * 1.1;

	PointCloud<Normal>::Ptr cloud_normals = estimateNormals(cloud,
			normalRadius);

	cout << "estimating normals... (this'll take a while)" << endl;
	PointCloud<PFHSignature125>::Ptr cloud_pfhs = estimatePfh(cloud,
			cloud_normals, pfhRadius);
	cout << "finished estimating normals!" << endl;
	cout << "writing output to file..." << endl;

	ofstream out;
	out.open("points_normals_pfhs.out", ios::out | ios::trunc | ios::binary);
	for (size_t i = 0; i < cloud_pfhs->size(); i++) {
		out << cloud->at(i) << " " << cloud_normals->at(i) << " "
				<< cloud_pfhs->at(i) << endl;
	}
	out.close();

	cout << "done, exiting!" << endl;

}
