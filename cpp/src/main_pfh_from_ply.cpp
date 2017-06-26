#include <algorithm>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"

#include "pcl/common/transforms.h"
#include "pcl/features/normal_3d.h"
#include "pcl/features/pfh.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "pcl/search/brute_force.h"
#include "pcl/search/kdtree.h"

using namespace pcl;
using namespace std;

string outfile;
bool save_as_pcd;
bool compute_incrementally;

/*
 * Point with position, surface normal estimate, and a 125-feature PFH
 * descriptor
 */
struct PointNormalPFH125 {
  PCL_ADD_POINT4D;   // adds x, y, z (stored in float[4])
  PCL_ADD_NORMAL4D;  // adds normal_x, normal_y, normal_z (stored in float[4])
  union {
    struct {
      float curvature;
    };
    float data_c[4];
  };

  float histogram[125];
  static int descriptorSize() { return 125; }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointNormalPFH125,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(
        float, curvature, curvature)(float[125], histogram, pfh));

/*
 * Extend NormalEstimation to provide some statistics about the computation
 * time.
 */
template <typename PointInT, typename PointOutT>
class NormalEstimationWithDebugInfo
    : public NormalEstimation<PointInT, PointOutT> {
  using Feature<PointInT, PointOutT>::indices_;
  using Feature<PointInT, PointOutT>::input_;
  using Feature<PointInT, PointOutT>::surface_;
  using Feature<PointInT, PointOutT>::k_;
  using Feature<PointInT, PointOutT>::search_parameter_;
  using NormalEstimation<PointInT, PointOutT>::vpx_;
  using NormalEstimation<PointInT, PointOutT>::vpy_;
  using NormalEstimation<PointInT, PointOutT>::vpz_;

  typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;

  void computeFeature(PointCloudOut& output) override {
    vector<int> nn_indices(k_);
    vector<float> nn_dists(k_);

    output.is_dense = true;
    // Save a few cycles by not checking every point for NaN/Inf values if the
    // cloud is set to dense
    if (input_->is_dense) {
      // Iterating over the entire index vector
      for (size_t idx = 0; idx < indices_->size(); ++idx) {
        if (this->searchForNeighbors((*indices_)[idx], search_parameter_,
                                     nn_indices, nn_dists) == 0) {
          output.points[idx].normal[0] = output.points[idx].normal[1] =
              output.points[idx].normal[2] = output.points[idx].curvature =
                  numeric_limits<float>::quiet_NaN();

          output.is_dense = false;
          printProgress(idx, nn_indices.size());
          continue;
        }

        this->computePointNormal(
            *surface_, nn_indices, output.points[idx].normal[0],
            output.points[idx].normal[1], output.points[idx].normal[2],
            output.points[idx].curvature);

        flipNormalTowardsViewpoint(input_->points[(*indices_)[idx]], vpx_, vpy_,
                                   vpz_, output.points[idx].normal[0],
                                   output.points[idx].normal[1],
                                   output.points[idx].normal[2]);

        printProgress(idx, nn_indices.size());
      }
    } else {
      // Iterating over the entire index vector
      for (size_t idx = 0; idx < indices_->size(); ++idx) {
        if (!isFinite((*input_)[(*indices_)[idx]]) ||
            this->searchForNeighbors((*indices_)[idx], search_parameter_,
                                     nn_indices, nn_dists) == 0) {
          output.points[idx].normal[0] = output.points[idx].normal[1] =
              output.points[idx].normal[2] = output.points[idx].curvature =
                  numeric_limits<float>::quiet_NaN();

          output.is_dense = false;
          printProgress(idx, nn_indices.size());
          continue;
        }

        this->computePointNormal(
            *surface_, nn_indices, output.points[idx].normal[0],
            output.points[idx].normal[1], output.points[idx].normal[2],
            output.points[idx].curvature);

        flipNormalTowardsViewpoint(input_->points[(*indices_)[idx]], vpx_, vpy_,
                                   vpz_, output.points[idx].normal[0],
                                   output.points[idx].normal[1],
                                   output.points[idx].normal[2]);

        printProgress(idx, nn_indices.size());
      }
    }
  }

 private:
  int64_t totalNNs = 0;
  int64_t fewestNNs = -1;
  int64_t mostNNs = -1;
  void printProgress(int current_idx, int num_nns) {
    totalNNs += num_nns;
    if (current_idx == 0) {
      fewestNNs = mostNNs = static_cast<int64_t>(num_nns);
    } else {
      fewestNNs = std::min(fewestNNs, static_cast<int64_t>(num_nns));
      mostNNs = std::max(mostNNs, static_cast<int64_t>(num_nns));
    }
    int total = indices_->size();
    if (current_idx % 1000 == 0 || current_idx == total - 1) {
      cout << static_cast<int>(100 * current_idx / total) << "% ("
           << current_idx << "/" << total << "), found " << num_nns
           << " nearest neighbors. Avg #nns = "
           << static_cast<double>(totalNNs) / (current_idx + 1)
           << ", min #nns = " << fewestNNs << ", max #nns = " << mostNNs
           << endl;
    }
  }
};

template <typename PointInT, typename PointNT,
          typename PointOutT = pcl::PFHSignature125>
class PFHEstimationWithDebugInfo
    : public PFHEstimation<PointInT, PointNT, PointOutT> {
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

  void computeFeature(PointCloudOut& output) override {
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
    // Save a few cycles by not checking every point for NaN/Inf values if the
    // cloud is set to dense
    if (input_->is_dense) {
      // Iterating over the entire index vector
      for (size_t idx = 0; idx < indices_->size(); ++idx) {
        if (this->searchForNeighbors((*indices_)[idx], search_parameter_,
                                     nn_indices, nn_dists) == 0) {
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
        if (!isFinite((*input_)[(*indices_)[idx]]) ||
            !isFinite((*normals_)[(*indices_)[idx]]) ||
            this->searchForNeighbors((*indices_)[idx], search_parameter_,
                                     nn_indices, nn_dists) == 0) {
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
  int64_t totalNNs = 0;
  int64_t fewestNNs = -1;
  int64_t mostNNs = -1;
  void printProgress(int current_idx, int num_nns) {
    totalNNs += num_nns;
    if (current_idx == 0) {
      fewestNNs = mostNNs = static_cast<int64_t>(num_nns);
    } else {
      fewestNNs = std::min(fewestNNs, static_cast<int64_t>(num_nns));
      mostNNs = std::max(mostNNs, static_cast<int64_t>(num_nns));
    }
    int total = indices_->size();
    if (current_idx % 1000 == 0 || current_idx == total - 1) {
      cout << static_cast<int>(100 * current_idx / total) << "% ("
           << current_idx << "/" << total << "), found " << num_nns
           << " nearest neighbors. Avg #nns = "
           << static_cast<double>(totalNNs) / (current_idx + 1)
           << ", min #nns = " << fewestNNs << ", max #nns = " << mostNNs
           << endl;
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
PointCloud<Normal>::Ptr estimateNormalsWindowedFrame(
    PointCloud<PointXYZI>::ConstPtr cloud, double search_radius) {
  // TODO: can use NormalEstimateOMP for drop-in multicore speedup
  NormalEstimation<PointXYZI, Normal> ne;
  ne.setInputCloud(cloud);

  // TODO: if we're only using a subset of the data, it might be better to
  // use a different search structure? maybe brute force exhaustive search?
  search::KdTree<PointXYZI>::Ptr tree(new search::KdTree<PointXYZI>());
  ne.setSearchMethod(tree);

  ne.setRadiusSearch(search_radius);

  PointCloud<Normal>::Ptr cloud_normals(new PointCloud<Normal>);

  ne.setViewPoint(cloud->sensor_origin_.x(), cloud->sensor_origin_.y(),
                  cloud->sensor_origin_.z());
  ne.compute(*cloud_normals);

  return cloud_normals;
}

PointCloud<PFHSignature125>::Ptr estimatePfh(
    PointCloud<PointXYZI>::Ptr cloud, PointCloud<Normal>::Ptr cloud_normals,
    double search_radius) {
  PFHEstimationWithDebugInfo<PointXYZI, Normal, PFHSignature125> pfh;
  pfh.setInputCloud(cloud);
  pfh.setInputNormals(cloud_normals);

  // Create an empty kdtree representation, and pass it to the PFH estimation
  // object.
  // Its content will be filled inside the object, based on the given input
  // dataset (as no other search surface is given).
  search::KdTree<PointXYZI>::Ptr tree(new search::KdTree<PointXYZI>());
  pfh.setSearchMethod(tree);

  // Output datasets
  PointCloud<PFHSignature125>::Ptr pfhs(new PointCloud<PFHSignature125>());

  // Use all neighbors in a sphere
  pfh.setRadiusSearch(search_radius);

  // Compute the features
  pfh.compute(*pfhs);

  return pfhs;
}
PointCloud<PFHSignature125>::Ptr estimatePfh(
    PointCloud<PointXYZI>::Ptr cloud, PointCloud<Normal>::Ptr cloud_normals,
    const IndicesConstPtr& indices, double search_radius) {
  PFHEstimation<PointXYZI, Normal, PFHSignature125> pfh;
  pfh.setInputCloud(cloud);
  pfh.setInputNormals(cloud_normals);
  pfh.setIndices(indices);

  // for small subsets of the data, brute force might be faster than rebuilding
  // the tree?
  // search::BruteForce<PointXYZI>::Ptr search_strategy(new
  // search::BruteForce<PointXYZI>());
  search::KdTree<PointXYZI>::Ptr search_strategy(
      new search::KdTree<PointXYZI>());
  pfh.setSearchMethod(search_strategy);

  // Output datasets
  PointCloud<PFHSignature125>::Ptr pfhs(new PointCloud<PFHSignature125>());

  // Use all neighbors in a sphere
  pfh.setRadiusSearch(search_radius);

  // Compute the features
  pfh.compute(*pfhs);

  return pfhs;
}

void saveResultsToFile(const PointCloud<PointXYZI>& cloud,
                       const PointCloud<Normal>& cloud_normals,
                       const PointCloud<PFHSignature125>& cloud_pfhs) {
  if (save_as_pcd) {
    PointCloud<PointNormalPFH125> cloud_combined;
    for (size_t i = 0; i < cloud_pfhs.size(); ++i) {
      PointNormalPFH125 point;
      copy(&cloud.at(i).data[0], &cloud.at(i).data[4], point.data);
      copy(&cloud_normals.at(i).data_n[0], &cloud_normals.at(i).data_n[4],
           point.data_n);
      copy(&cloud_normals.at(i).data_c[0], &cloud_normals.at(i).data_c[4],
           point.data_c);
      copy(&cloud_pfhs.at(i).histogram[0],
           &cloud_pfhs.at(i).histogram[PFHSignature125::descriptorSize()],
           point.histogram);

      cloud_combined.push_back(point);
    }
    io::savePCDFileBinary(outfile, cloud_combined);
  } else {
    ofstream out;
    out.open(outfile, ios::out | ios::trunc | ios::binary);
    for (size_t i = 0; i < cloud_pfhs.size(); i++) {
      out << cloud.at(i) << " " << cloud_normals.at(i) << " "
          << cloud_pfhs.at(i) << endl;
    }
    out.close();
    cout << "wrote " << cloud_pfhs.size() << " lines!" << endl;
  }
}

/*
 * This computes normals, using only points located in the last window_size
 * keyframes before the current one one, and the first window_size after it.
 * Points in the first and last window_size keyframes are ignored.
 */
int doIncrementalEstimate(const string& filepattern, int window_size,
                          double normalRadius, double pfhRadius) {
  PointCloud<PointXYZI>::Ptr cumulative_cloud(new PointCloud<PointXYZI>);
  list<PointCloud<PointXYZI>::Ptr> clouds_in_window;

  // load the first few without computing normals
  for (int i = 0; i <= 2 * window_size; ++i) {
    PointCloud<PointXYZI>::Ptr cur_cloud(new PointCloud<PointXYZI>);
    stringstream filename;
    filename << filepattern << "_" << i << ".pcd";
    // load the file
    if (io::loadPCDFile<PointXYZI>(filename.str(), *cur_cloud) == -1) {
      PCL_ERROR("Couldn't read file %s \n", filepattern);
      return -1;
    }
    transformPointCloud(*cur_cloud, *cur_cloud,
                        cur_cloud->sensor_origin_.head<3>(),
                        cur_cloud->sensor_orientation_);

    cumulative_cloud->points.insert(end(cumulative_cloud->points),
                                    begin(cur_cloud->points),
                                    end(cur_cloud->points));
    clouds_in_window.push_back(cur_cloud);
  }

  int i = window_size;
  size_t points_before = 0;
  auto cloud_iter = clouds_in_window.begin();
  for (int j = 0; j < window_size; ++j, ++cloud_iter) {
    points_before += (*cloud_iter)->size();
  }

  PointCloud<PointXYZI> result_points;
  PointCloud<Normal> result_normals;
  PointCloud<PFHSignature125> result_pfhs;

  while (true) {
    if (i % 10 == 0) {
      cout << "processing frame " << i << endl;
    }
    size_t current_num_points = (*cloud_iter)->size();
    IndicesPtr indices(new vector<int>(current_num_points));
    for (size_t j = 0; j < current_num_points; j++) {
      indices->at(j) = j + points_before;
    }

    // cout << "estimating normals..." << endl;
    // compute normals for the whole windowed cloud (need all of them even
    // to compute pfh at only a subset)
    PointCloud<Normal>::Ptr cloud_normals =
        estimateNormalsWindowedFrame(cumulative_cloud, normalRadius);

    // cout << "estimating pfh..." << endl;
    // only compute pfh at current indices
    PointCloud<PFHSignature125>::Ptr cloud_pfhs =
        estimatePfh(cumulative_cloud, cloud_normals, indices, pfhRadius);

    // cout << "finished estimated, performing bookkeeping..." << endl;

    // store results
    result_points.insert(begin(result_points), begin((*cloud_iter)->points),
                         end((*cloud_iter)->points));
    // only select the normals in the middle frame
    result_normals.insert(
        begin(result_normals), &cloud_normals->points.begin()[points_before],
        &cloud_normals->points.begin()[points_before + current_num_points]);
    result_pfhs.insert(begin(result_pfhs), begin(cloud_pfhs->points),
                       end(cloud_pfhs->points));

    // remove points from frame i-window_size
    auto front = begin(cumulative_cloud->points);
    size_t num_to_remove = clouds_in_window.front()->size();
    cumulative_cloud->points.erase(front, front + num_to_remove);
    points_before -= num_to_remove;
    clouds_in_window.pop_front();

    // add points in frame i+1+window_size
    PointCloud<PointXYZI>::Ptr next_cloud(new PointCloud<PointXYZI>);
    stringstream filename;
    filename << filepattern << "_" << (i + 1 + window_size) << ".pcd";

    if (!boost::filesystem::exists(filename.str())) {
      // TODO: scan the whole directory explicitly ahead of time
      break;
    }
    if (io::loadPCDFile<PointXYZI>(filename.str(), *next_cloud) == -1) {
      PCL_ERROR("Couldn't read file %s \n", filepattern);
      return -1;
    }
    transformPointCloud(*next_cloud, *next_cloud,
                        next_cloud->sensor_origin_.head<3>(),
                        next_cloud->sensor_orientation_);

    points_before += current_num_points;
    cumulative_cloud->points.insert(end(cumulative_cloud->points),
                                    begin(next_cloud->points),
                                    end(next_cloud->points));
    clouds_in_window.push_back(next_cloud);
    ++i;
    ++cloud_iter;
  }

  saveResultsToFile(result_points, result_normals, result_pfhs);

  return 0;
}

int doFullEstimate(const string& filepattern, double normalRadius,
                   double pfhRadius) {
  PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);

  // load the file
  if (io::loadPCDFile<PointXYZI>(filepattern, *cloud) == -1) {
    PCL_ERROR("Couldn't read file %s \n", filepattern);
    return -1;
  }
  cout << "Loaded " << cloud->width * cloud->height << " data points from "
       << filepattern << endl;

  // the flann kd search tree is somehow built incrementally based on the
  // pointcloud index ordering. randomize the order, otherwise the tree will
  // be very slow for points far from the initial camera position
  random_shuffle(cloud->begin(), cloud->end());

  PointCloud<Normal>::Ptr cloud_normals = estimateNormals(cloud, normalRadius);

  cout << "estimating pfh... (this'll take a while)" << endl;
  PointCloud<PFHSignature125>::Ptr cloud_pfhs =
      estimatePfh(cloud, cloud_normals, pfhRadius);
  cout << "finished estimating pfh!" << endl;
  cout << "writing output to file..." << endl;

  saveResultsToFile(*cloud, *cloud_normals, *cloud_pfhs);

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
  // note: the normal computation search radius must be smaller than the PFH
  // search radius
  double normalRadius = 0.05;
  double pfhRadius = normalRadius * 1.1;

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
    return doIncrementalEstimate(filepattern, 10, normalRadius, pfhRadius);
  } else {
    return doFullEstimate(filepattern, normalRadius, pfhRadius);
  }
}
