#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>

using namespace pcl;
using namespace std;

string outfile;
bool save_as_pcd;
bool compute_incrementally;

/*
 * Point with position, surface normal estimate, and a 125-feature PFH descriptor
 */
struct PointNormalPFH125 {
	PCL_ADD_POINT4D
	; // adds x, y, z (stored in float[4])
	PCL_ADD_NORMAL4D
	; // adds normal_x, normal_y, normal_z (stored in float[4])
	union {
		struct {
			float curvature;
		};
		float data_c[4];
	};

	float histogram[125];
	static int descriptorSize() {
		return 125;
	}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointNormalPFH125,
		(float, x, x) (float, y, y) (float, z, z) (float, normal_x, normal_x) (float, normal_y, normal_y) (float, normal_z, normal_z) (float, curvature, curvature) (float[125], histogram, pfh));

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

/*
 * This computes normals, using only points located in the last window_size
 * keyframes before the current one one, and the first window_size after it.
 * Points in the first and last window_size keyframes are ignored.
 */
int doIncrementalEstimate(const string& filepattern, int window_size) {


	return 0;
}

int doFullEstimate(const string& filepattern) {
	PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);

	if (io::loadPCDFile<PointXYZI>(filepattern, *cloud) == -1) //* load the file
			{
		PCL_ERROR("Couldn't read file %s \n", filepattern);
		return -1;
	}
	cout << "Loaded " << cloud->width * cloud->height << " data points from "
			<< filepattern << endl;

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

	if (save_as_pcd) {
		PointCloud<PointNormalPFH125> cloud_combined;
		for (size_t i = 0; i < cloud_pfhs->size(); ++i) {
			PointNormalPFH125 point;
			copy(&cloud->at(i).data[0], &cloud->at(i).data[4], point.data);
			copy(&cloud_normals->at(i).data_n[0],
					&cloud_normals->at(i).data_n[4], point.data_n);
			copy(&cloud_normals->at(i).data_c[0],
					&cloud_normals->at(i).data_c[4], point.data_c);
			copy(&cloud_pfhs->at(i).histogram[0],
					&cloud_pfhs->at(i).histogram[PFHSignature125::descriptorSize()],
					point.histogram);

			cloud_combined.push_back(point);
		}
		io::savePCDFileBinary(outfile, cloud_combined);
	} else {
		ofstream out;
		out.open(outfile, ios::out | ios::trunc | ios::binary);
		for (size_t i = 0; i < cloud_pfhs->size(); i++) {
			out << cloud->at(i) << " " << cloud_normals->at(i) << " "
					<< cloud_pfhs->at(i) << endl;
		}
		out.close();
		cout << "wrote " << cloud_pfhs->size() << " lines!" << endl;
	}

	cout << "done, exiting!" << endl;
	return 0;
}

void parseArgument(char* arg) {
	int option;
	char buf[1000];

	if (1 == sscanf(arg, "pcd_out=%s", buf)) {
		outfile = string(buf);
		printf("saving pcd file to %s\n", outfile.c_str());
		save_as_pcd = true;
		return;
	}

	if (1 == sscanf(arg, "txt_out=%s", buf)) {
		outfile = string(buf);
		printf("saving ascii text file to %s\n", outfile.c_str());
		save_as_pcd = false;
		return;
	}

	if (1 == sscanf(arg, "incremental=%d", &option)) {
		if (option == 1) {
			printf("computing normals incrementally\n");
			compute_incrementally = true;
		} else {
			printf("computing normals on the full pointcloud\n");
			compute_incrementally = false;
		}
		return;
	}
	printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char** argv) {

	if (argc < 4) {
		printf("too few arguments!\n");
		// TODO: print usage
		return 1;
	}

	string filepattern(argv[1]);
	for (int i = 2; i < argc; i++) {
		parseArgument(argv[i]);
	}

	if (compute_incrementally) {
		return doIncrementalEstimate(filepattern, 10);
	} else {
		return doFullEstimate(filepattern);
	}
}
