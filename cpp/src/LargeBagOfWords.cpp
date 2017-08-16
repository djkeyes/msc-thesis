#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "opencv2/flann/miniflann.hpp"

#include "LargeBagOfWords.h"

using std::vector;

namespace cv {

const int kdtree_numtrees = 8;
const int kdtree_numchecks = 768;

static void generateRandomCenter(const std::vector<Vec2f>& box, float* center,
                                 RNG& rng) {
  size_t j, dims = box.size();
  float margin = 1.f / dims;
  for (j = 0; j < dims; j++)
    center[j] = (static_cast<float>(rng) * (1.f + margin * 2.f) - margin) *
                    (box[j][1] - box[j][0]) +
                box[j][0];
}

static void generateCentersPP(const Mat& _data, Mat& _out_centers, int K,
                              RNG& rng, int trials) {
  throw std::logic_error(
      "KMeans++ is not implemented for ApproxKMeans! "
      "Running it would be too slow (O(N*K*D), unless you can think of "
      "a good approximation scheme).");
  // We might be able to do this in O(N * log(K)^2 * D). Instead of sampling
  // K points and updating the distances each time, we could sample
  // 1, 2, 4, ... 2^i points, and then store the new centers in a KD-tree or
  // ANN tree. Bookkeeping would be tricky though.
}
class ApproxKMeansDistanceComputer : public ParallelLoopBody {
 public:
  ApproxKMeansDistanceComputer(double* _distances, int* _labels,
                               const Mat& _data, flann::Index& _flann_index,
                               bool _onlyDistance = false)
      : distances(_distances),
        labels(_labels),
        data(_data),
        flann_index(_flann_index) {}

  void operator()(const Range& range) const {
    Mat indices(range.size(), 1, CV_32S, &labels[range.start]);
    Mat dists(range.size(), 1, CV_32F, &distances[range.start]);

    flann_index.knnSearch(data.rowRange(range), indices, dists, 1,
                          flann::SearchParams(kdtree_numchecks, /*eps=*/0.0));
  }

 private:
  double* distances;
  int* labels;
  const Mat& data;
  flann::Index& flann_index;
};

double approx_kmeans(InputArray _data, int K, InputOutputArray _bestLabels,
                     TermCriteria criteria, int attempts, int flags,
                     OutputArray _centers) {
  const int SPP_TRIALS = 3;
  Mat data0 = _data.getMat();
  bool isrow = data0.rows == 1;
  int N = isrow ? data0.cols : data0.rows;
  int dims = (isrow ? 1 : data0.cols) * data0.channels();
  int type = data0.depth();

  attempts = std::max(attempts, 1);
  CV_Assert(data0.dims <= 2 && type == CV_32F && K > 0);
  // TODO: if N < K, we could just pad with 0-centered clusters
  CV_Assert(N >= K);

  Mat data(N, dims, CV_32F, data0.ptr(),
           isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

  _bestLabels.create(N, 1, CV_32S, -1, true);

  Mat _labels, best_labels = _bestLabels.getMat();
  if (flags & CV_KMEANS_USE_INITIAL_LABELS) {
    CV_Assert((best_labels.cols == 1 || best_labels.rows == 1) &&
              best_labels.cols * best_labels.rows == N &&
              best_labels.type() == CV_32S && best_labels.isContinuous());
    best_labels.copyTo(_labels);
  } else {
    if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
          best_labels.cols * best_labels.rows == N &&
          best_labels.type() == CV_32S && best_labels.isContinuous()))
      best_labels.create(N, 1, CV_32S);
    _labels.create(best_labels.size(), best_labels.type());
  }
  int* labels = _labels.ptr<int>();

  Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
  std::vector<int> counters(K);
  std::vector<Vec2f> _box(dims);
  Mat dists(1, N, CV_64F);
  Vec2f* box = &_box[0];
  double best_compactness = DBL_MAX, compactness = 0;
  RNG& rng = theRNG();
  int a, iter, i, j, k;

  if (criteria.type & TermCriteria::EPS)
    criteria.epsilon = std::max(criteria.epsilon, 0.);
  else
    criteria.epsilon = FLT_EPSILON;
  criteria.epsilon *= criteria.epsilon;

  if (criteria.type & TermCriteria::COUNT)
    criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
  else
    criteria.maxCount = 100;

  if (K == 1) {
    attempts = 1;
    criteria.maxCount = 2;
  }

  const float* sample = data.ptr<float>(0);
  for (j = 0; j < dims; j++) box[j] = Vec2f(sample[j], sample[j]);

  for (i = 1; i < N; i++) {
    sample = data.ptr<float>(i);
    for (j = 0; j < dims; j++) {
      float v = sample[j];
      box[j][0] = std::min(box[j][0], v);
      box[j][1] = std::max(box[j][1], v);
    }
  }

  for (a = 0; a < attempts; a++) {
    double max_center_shift = DBL_MAX;
    for (iter = 0;;) {
      std::cout << "iter=" << iter << std::endl;
      swap(centers, old_centers);

      if (iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS))) {
        std::cout << "initial" << std::endl;
        if (flags & KMEANS_PP_CENTERS) {
          generateCentersPP(data, centers, K, rng, SPP_TRIALS);
        } else {
          for (k = 0; k < K; k++)
            generateRandomCenter(_box, centers.ptr<float>(k), rng);
        }
      } else {
        if (iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS)) {
          for (i = 0; i < N; i++) CV_Assert((unsigned)labels[i] < (unsigned)K);
        }

        // compute centers
        centers = Scalar(0);
        for (k = 0; k < K; k++) {
          counters[k] = 0;
        }

        for (i = 0; i < N; i++) {
          sample = data.ptr<float>(i);
          k = labels[i];
          float* center = centers.ptr<float>(k);
          for (j = 0; j < dims; j++) {
            center[j] += sample[j];
          }
          counters[k]++;
        }

        if (iter > 0) max_center_shift = 0;

        int num_empty_clusters = 0;
        for (k = 0; k < K; k++) {
          if (counters[k] == 0) {
            num_empty_clusters++;
          }
        }
        if (num_empty_clusters > 0) {
          std::cout << num_empty_clusters << " empty clusters!" << std::endl;
          // if some cluster appeared to be empty then:
          //   1. find the biggest cluster
          //   2. find the farthest from the center point in the biggest cluster
          //   3. exclude the farthest point from the biggest cluster and form a
          //   new 1-point cluster.
          // To this end:
          //   1. put clusters in a heap sorted by size, to efficiently fetch
          //   max element. We only need the top-most heaps.
          //   2. create a smaller heap for each largest cluster, sorted by
          //   distance
          //   3. for each point, add it to the heap of its cluster if it's in
          //   one of the larger clusters

          auto comp = [&counters](const int& first, const int& second) {
            return counters[first] - counters[second] <= 0;
          };
          std::priority_queue<int, std::vector<int>, decltype(comp)>
              cluster_size_queue(comp);
          for (int k = 0; k < K; k++) {
            if (counters[k] > 0) {
              cluster_size_queue.push(k);
            }
          }

          std::list<int> largest_clusters;
          int num_removable_points = 0;
          int next_count;
          while (num_removable_points < num_empty_clusters) {
            int last = cluster_size_queue.top();
            largest_clusters.push_back(last);
            cluster_size_queue.pop();
            if (cluster_size_queue.empty()) {
              next_count = 0;
            } else {
              next_count = counters[cluster_size_queue.top()];
            }
            num_removable_points +=
                largest_clusters.size() * (counters[last] - next_count);
          }
          unsigned int num_points_to_leave =
              (num_removable_points - num_empty_clusters) /
                  largest_clusters.size() +
              next_count;

          auto point_comp = [&dists](const int& first, const int& second) {
            return dists.at<double>(first) - dists.at<double>(second) <= 0;
          };
          std::map<int, std::priority_queue<int, std::vector<int>,
                                            decltype(point_comp)>>
              removable_points;
          for (const auto k : largest_clusters) {
            // need to use fancy emplace semantics, since lambda
            // have no default constructor
            removable_points.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(k),
                                     std::forward_as_tuple(point_comp));
          }
          for (int i = 0; i < N; i++) {
            auto target_queue = removable_points.find(labels[i]);
            if (target_queue != removable_points.end()) {
              target_queue->second.push(i);
            }
          }

          auto candidate_point_iter = removable_points.begin();
          for (int k = 0; k < K; k++) {
            if (counters[k] != 0) continue;

            if (candidate_point_iter->second.size() == num_points_to_leave) {
              ++candidate_point_iter;
            }

            int max_k = candidate_point_iter->first;

            int farthest_i = candidate_point_iter->second.top();
            // this uses the old distance instead of calculating new ones
            candidate_point_iter->second.pop();

            float* new_center = centers.ptr<float>(k);
            float* old_center = centers.ptr<float>(max_k);

            counters[max_k]--;
            counters[k]++;
            labels[farthest_i] = k;
            sample = data.ptr<float>(farthest_i);

            for (j = 0; j < dims; j++) {
              old_center[j] -= sample[j];
              new_center[j] += sample[j];
            }
          }
        }

        // Normalize centroids, and compute convergence checks
        for (k = 0; k < K; k++) {
          float* center = centers.ptr<float>(k);
          CV_Assert(counters[k] != 0);

          float scale = 1.f / counters[k];
          for (j = 0; j < dims; j++) {
            center[j] *= scale;
          }

          if (iter > 0) {
            double dist = 0;
            const float* old_center = old_centers.ptr<float>(k);
            for (j = 0; j < dims; j++) {
              double t = center[j] - old_center[j];
              dist += t * t;
            }
            max_center_shift = std::max(max_center_shift, dist);
          }
        }
      }

      bool isLastIter = (++iter == MAX(criteria.maxCount, 2) ||
                         max_center_shift <= criteria.epsilon);

      flann::KDTreeIndexParams params(kdtree_numtrees);
      flann::Index flann_index(centers, params);

      // assign labels
      dists = 0;
      double* dist = dists.ptr<double>(0);
      // TODO: is FLANN already multithreaded? If so, this might just be adding
      // overhead
      parallel_for_(Range(0, N),
                    ApproxKMeansDistanceComputer(dist, labels, data,
                                                 flann_index, isLastIter));
      compactness = sum(dists)[0];

      std::cout << "total squared distance: " << compactness << std::endl;

      if (isLastIter) break;
    }

    if (compactness < best_compactness) {
      best_compactness = compactness;
      if (_centers.needed()) centers.copyTo(_centers);
      _labels.copyTo(best_labels);
    }
  }

  return best_compactness;
}

BOWApproxKMeansTrainer::BOWApproxKMeansTrainer(int _clusterCount,
                                               const TermCriteria& _termcrit,
                                               int _attempts, int _flags)
    : BOWKMeansTrainer(_clusterCount, _termcrit, _attempts, _flags) {}

Mat BOWApproxKMeansTrainer::cluster() const {
  CV_Assert(!descriptors.empty());

  Mat mergedDescriptors(descriptorsCount(), descriptors[0].cols,
                        descriptors[0].type());
  for (size_t i = 0, start = 0; i < descriptors.size(); i++) {
    Mat submut = mergedDescriptors.rowRange(
        static_cast<int>(start), static_cast<int>(start + descriptors[i].rows));
    descriptors[i].copyTo(submut);
    start += descriptors[i].rows;
  }
  return cluster(mergedDescriptors);
}

BOWApproxKMeansTrainer::~BOWApproxKMeansTrainer() {}

Mat BOWApproxKMeansTrainer::cluster(const Mat& _descriptors) const {
  Mat labels, vocabulary;
  approx_kmeans(_descriptors, clusterCount, labels, termcrit, attempts, flags,
                vocabulary);
  return vocabulary;
}

CV_WRAP BOWSparseImgDescriptorExtractor::BOWSparseImgDescriptorExtractor(
    const Ptr<DescriptorExtractor>& dextractor,
    const Ptr<DescriptorMatcher>& dmatcher)
    : BOWImgDescriptorExtractor(dextractor, dmatcher) {}
BOWSparseImgDescriptorExtractor::~BOWSparseImgDescriptorExtractor() {}

void BOWSparseImgDescriptorExtractor::computeAssignments(
    InputArray keypointDescriptors, vector<vector<int>>& assignmentsOut,
    int num_nearest_neighbors) {
  CV_Assert(!vocabulary.empty());

  // Match keypoint descriptors to cluster center (to vocabulary)
  vector<vector<DMatch>> matches;
  dmatcher->knnMatch(keypointDescriptors, matches, num_nearest_neighbors);

  assignmentsOut.reserve(matches.size());

  for (size_t i = 0; i < matches.size(); ++i) {
    assignmentsOut.emplace_back();
    assignmentsOut.back().reserve(matches[i].size());
    for (size_t j = 0; j < matches[i].size(); ++j) {
      int queryIdx = matches[i][j].queryIdx;
      int trainIdx = matches[i][j].trainIdx;
      CV_Assert(queryIdx == static_cast<int>(i));

      assignmentsOut.back().push_back(trainIdx);
    }
  }
}
}  // namespace cv
