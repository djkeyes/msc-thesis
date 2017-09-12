#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "caffe/caffe.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/operations.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "Datasets.h"
#include "FusedFeatureDescriptors.h"
#include "Relocalization.h"
#include "CaffeDescriptor.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

namespace sdl {

bool use_second_best_for_debugging = false;
bool save_first_trajectory = true;
bool display_top_matching_images = true;
bool display_stereo_correspondences = true;
bool use_5pt_for_verif = false;  // true: 5pt essential mat. false: 4pt PnP

int ransac_num_iters = 10000;  // 500;
double ransac_confidence = 0.999999;  // 0.999;
double ransac_threshold = 8.0;  // 8.0;
int ransac_batch_size = 500;

int vocabulary_size;
double epsilon_angle_deg;
double epsilon_translation;
double ratio_threshold;

bool use_caffe_descriptor;
string caffe_prototxt;
string caffe_caffemodel;

bool seed_ground_truth_poses;

unique_ptr<SceneParser> scene_parser;

void solvePnP(const vector<cv::Point3f>& scene_points,
              const vector<cv::Point2f>& query_points, Mat K, Mat& rvec, Mat& t,
              bool use_iterative_refinement) {
  cv::solvePnP(scene_points, query_points, K, noArray(), rvec, t, false,
               cv::SOLVEPNP_EPNP);
  if (use_iterative_refinement) {
    cv::solvePnP(scene_points, query_points, K, noArray(), rvec, t, true,
                 cv::SOLVEPNP_ITERATIVE);
  }
}
void solvePnPRansac(const vector<cv::Point3f>& scene_points,
                    const vector<cv::Point2f>& query_points, Mat K, Mat& rvec,
                    Mat& t, int max_iters, float threshold, float confidence,
                    Mat& inliers, bool use_iterative_refinement) {
  int flag = SOLVEPNP_EPNP;
  if (use_iterative_refinement) {
    // this only applies to the final pose. the initial guesses from ransac are
    // all estimated with EPnP.
    flag = SOLVEPNP_ITERATIVE;
  }
  cv::solvePnPRansac(scene_points, query_points, K, noArray(), rvec, t, false,
                     max_iters, threshold, confidence, inliers, flag);
}
// Perform RANSAC on incrementally larger subsets of the correspondences, as in
// Schmidt et al.
void solvePreemptivePnPRansac(const vector<cv::Point3f>& scene_points,
                              const vector<cv::Point2f>& query_points, Mat K,
                              Mat& _rvec, Mat& _t, int num_hypotheses, int batch_size, float threshold,
                              Mat& _inliers, bool use_iterative_refinement) {
  std::mt19937 rng;

  vector<int> indices(scene_points.size());
  for (int i = 0; i < static_cast<int>(scene_points.size()); ++i) {
    indices[i] = i;
  }
  std::shuffle(indices.begin(), indices.end(), rng);

  // Note: in Schmidt et al., they specifically enforced a 1-1 correspondence by
  // picking the correspondence with smallest reprojection error during RANSAC.
  // We could do this, too, by checking which query_points are identical--or by
  // requiring users to pass us a map<int, list<int>> instead.

  // sample some initial hypotheses
  // <rvec, t, inliers>
  vector<tuple<Mat, Mat, Mat>> hypotheses;
  hypotheses.reserve(num_hypotheses);
  std::uniform_int_distribution<int> uniform(0, indices.size() - 1);
  for (int i = 0; i < num_hypotheses; ++i) {
    // it would be nice to do some kind of adaptation of a fisher-yates shuffle
    // but this is still fine for small n
    set<int> sample_indices;
    while (sample_indices.size() < 5) {
      int idx = uniform(rng);
      if (sample_indices.find(idx) != sample_indices.end()) {
        continue;
      }
      sample_indices.insert(idx);
    }
    vector<cv::Point3f> minimal_scene_points;
    vector<cv::Point2f> minimal_query_points;
    for (int j : sample_indices) {
      minimal_scene_points.push_back(scene_points[j]);
      minimal_query_points.push_back(query_points[j]);
    }
    Mat rvec(3, 1, CV_32F);
    Mat t(3, 1, CV_32F);
    solvePnP(minimal_scene_points, minimal_query_points, K, rvec, t, false);
//    solvePnP(minimal_scene_points, minimal_query_points, K, rvec, t, true);

    // for good measure, also add these points to the inliers
    Mat inliers(0, 1, CV_32S);
    for (auto index : sample_indices) {
      inliers.push_back(index);
    }
    hypotheses.push_back(make_tuple(rvec, t, inliers));
  }

  int offset = 0;
  while (hypotheses.size() > 1) {
    vector<cv::Point3f> next_sampled_scene_points;
    vector<cv::Point2f> next_sampled_query_points;
    next_sampled_scene_points.reserve(batch_size);
    next_sampled_query_points.reserve(batch_size);
    vector<int> orig_idx;
    for (int i = 0; i < batch_size; ++i) {
      if (offset + i >= static_cast<int>(indices.size())) {
        break;
      }
      int shuffled_idx = indices[offset + i];
      next_sampled_scene_points.push_back(scene_points[shuffled_idx]);
      next_sampled_query_points.push_back(query_points[shuffled_idx]);
      orig_idx.push_back(offset + i);
    }
    offset += batch_size;

    // only check for new inliers if we haven't run out of samples so far
    if (next_sampled_query_points.size() > 0) {
      // count new inliers
      for (unsigned int i = 0; i < hypotheses.size(); ++i) {
        Mat rvec = std::get<0>(hypotheses[i]);
        Mat t = std::get<1>(hypotheses[i]);
        Mat inliers = std::get<2>(hypotheses[i]);
        vector<cv::Point2f> projected;
        projected.reserve(next_sampled_scene_points.size());
        projectPoints(next_sampled_scene_points, rvec, t, K, noArray(),
                      projected);

        for (unsigned int j = 0; j < next_sampled_query_points.size(); ++j) {
          double error = cv::norm(projected[j] - next_sampled_query_points[j]);
          if (error < threshold) {
            inliers.push_back(orig_idx[j]);
          }
        }
        std::get<0>(hypotheses[i]) = rvec;
        std::get<1>(hypotheses[i]) = t;
        std::get<2>(hypotheses[i]) = inliers;
      }
    }

    // retain best half
    std::sort(hypotheses.begin(), hypotheses.end(),
              [](const tuple<Mat, Mat, Mat>& first,
                 const tuple<Mat, Mat, Mat>& second) {
                return std::get<2>(first).rows > std::get<2>(second).rows;
              });
//    cout << "inlier counts: ";
//    for (auto& element : hypotheses) {
//      cout << std::get<2>(element).rows << ",";
//    }
//    cout << endl;
    int next_num_hypotheses = hypotheses.size() / 2;
    hypotheses.resize(next_num_hypotheses);

    // refine hypotheses
    for (unsigned int i = 0; i < hypotheses.size(); ++i) {
      // this is potentially sort of slow, since we have to rebuild the inlier
      // sets for each hypotheses. It would be nice if we could slice the
      // original set somehow.
      Mat inliers = std::get<2>(hypotheses[i]);
      vector<cv::Point3f> scene_inliers;
      vector<cv::Point2f> query_inliers;
      int num_inliers = inliers.rows;
      scene_inliers.reserve(num_inliers);
      query_inliers.reserve(num_inliers);
      for (int j = 0; j < num_inliers; ++j) {
        scene_inliers.push_back(scene_points[inliers.at<int>(j)]);
        query_inliers.push_back(query_points[inliers.at<int>(j)]);
      }
      Mat rvec(3, 1, CV_32F);
      Mat t(3, 1, CV_32F);
      solvePnP(scene_inliers, query_inliers, K, rvec, t, false);
      std::get<0>(hypotheses[i]) = rvec;
      std::get<1>(hypotheses[i]) = t;
    }
  }

  _rvec = std::get<0>(hypotheses[0]);
  _t = std::get<1>(hypotheses[0]);
  _inliers = std::get<2>(hypotheses[0]);

  if (use_iterative_refinement) {
    // TODO: do one final optimization using the full set
    vector<cv::Point2f> projected;
    projectPoints(scene_points, _rvec, _t, K, noArray(), projected);

    _inliers = Mat();
    vector<cv::Point3f> scene_inliers;
    vector<cv::Point2f> query_inliers;
    for (unsigned int j = 0; j < query_points.size(); ++j) {
      double error = cv::norm(projected[j] - query_points[j]);
      if (error < threshold) {
        _inliers.push_back(static_cast<int>(j));
        scene_inliers.push_back(scene_points[j]);
        query_inliers.push_back(query_points[j]);
      }
    }
    cout << "final num inliers: " << scene_inliers.size() << endl;
    solvePnP(scene_inliers, query_inliers, K, _rvec, _t, true);
  }
}
class MatchingMethod {
 public:
  virtual ~MatchingMethod() = default;

  virtual bool needs3dDatabasePoints() = 0;
  virtual bool needs3dQueryPoints() = 0;

  void doMatching(const Query& q, const Result& top_result,
                  const vector<Point2f>& query_pts,
                  const vector<Point2f>& database_pts) {
    Mat R, t, inlier_mask;
    internalDoMatching(top_result, query_pts, database_pts, inlier_mask, R, t);

    updateResults(q, top_result, inlier_mask, R, t, query_pts, database_pts);
  }

  void setK(Mat K_) { K = K_; }

  void updateResults(const Query& q, const Result& top_result, Mat inlier_mask,
                     Mat R, Mat t, const vector<Point2f>& query_pts,
                     const vector<Point2f>& database_pts) {
    // should this threshold be a function of the ransac model DOF?
    if (countNonZero(inlier_mask) >= 12) {
      high_inlier_queries++;
    }

    if (countNonZero(inlier_mask) == 0) {
      // Pose estimation failed. Just use the db's VO pose as a guess.
      // If anyone wants to find failed poses later, they can check the inlier
      // count.
      int rt = R.type();
      int tt = t.type();
      dso::SE3 pose(top_result.frame.loadPose());
      R.create(3, 3, CV_64F);
      t.create(3, 1, CV_64F);
      eigen2cv(pose.rotationMatrix(), R);
      eigen2cv(pose.translation(), t);
      R.convertTo(R, rt);
      t.convertTo(t, tt);
    }

    float inlier_fraction = static_cast<float>(countNonZero(inlier_mask)) / (inlier_mask.rows * inlier_mask.cols);
    inlierFractionSum += inlier_fraction;
    inlierFractionCount++;
    cout << "latest inlier fraction: " << inlier_fraction << endl;
    cout << "average inlier fraction: " << (inlierFractionSum / inlierFractionCount) << endl;

    static int num_displayed = 0;
    if (display_stereo_correspondences && num_displayed < 20 &&
        q.getParentDatabaseId() !=
            static_cast<unsigned int>(top_result.frame.dbId) && q.getFrame()->index > 200) {
      Mat query = q.getFrame()->imageLoader();
      Mat result = top_result.frame.imageLoader();
      Mat stereo;
      stereo.create(query.rows, query.cols + result.cols, query.type());

      query.copyTo(stereo.colRange(0, query.cols));
      result.copyTo(stereo.colRange(query.cols, query.cols + result.cols));
      Point2f offset(query.cols, 0);
      for (int i = 0; i < inlier_mask.rows; ++i) {
        Scalar color;
        if (inlier_mask.at<int32_t>(i, 0) > 0) {
          color = Scalar(0, 255, 0);
        } else {
//          color = Scalar(0, 0, 255);
          continue;
        }

        line(stereo, query_pts[i], database_pts[i] + offset, color);
      }

      stringstream ss;
      ss << "Stereo correspondences for query " << q.getFrame()->index << "(db "
         << q.getFrame()->dbId << ") <==> result " << top_result.frame.index
         << " (db " << top_result.frame.dbId << ")";
      string window_name(ss.str());
      namedWindow(window_name, WINDOW_AUTOSIZE);
      imshow(window_name, stereo);
      waitKey(0);
      destroyWindow(window_name);

      Mat db_R_gt, db_t_gt;
      scene_parser->loadGroundTruthPose(top_result.frame, db_R_gt, db_t_gt);
      Mat query_R_gt, query_t_gt;
      scene_parser->loadGroundTruthPose(*q.getFrame(), query_R_gt, query_t_gt);

      double translation_error = norm(query_t_gt, t);
      // compute angle between rotation matrices
      Mat rotation_diff = (query_R_gt * R.t());
      // To measure difference, compute the angle when represented in axis-angle
      float angle_error_rad =
          acos((trace(rotation_diff)[0] - 1.0) /
               2.0);
      float angle_error_deg = (180. * angle_error_rad / M_PI);

      cout << "Error (only valid if DB is aligned to GT): " << angle_error_deg
           << " deg, " << translation_error << "m" << endl;

      ++num_displayed;
    }

    Mat db_R_gt, db_t_gt;
    scene_parser->loadGroundTruthPose(top_result.frame, db_R_gt, db_t_gt);
    Mat query_R_gt, query_t_gt;
    scene_parser->loadGroundTruthPose(*q.getFrame(), query_R_gt, query_t_gt);

    int train_id = top_result.frame.dbId;
    auto& db_gt_traj = groundTruthTrajectories[train_id];
    auto& db_vo_traj = voTrajectories[train_id];
    if (db_gt_traj.find(top_result.frame.index) == db_gt_traj.end()) {
      db_gt_traj[top_result.frame.index] =
          make_pair(db_R_gt.clone(), db_t_gt.clone());

      dso::SE3 pose(top_result.frame.loadPose());
      Mat db_R_vo(3, 3, CV_64F), db_t_vo(3, 1, CV_64F);
      eigen2cv(pose.rotationMatrix(), db_R_vo);
      eigen2cv(pose.translation(), db_t_vo);

      db_vo_traj[top_result.frame.index] = make_pair(db_R_vo, db_t_vo);
    }

    int test_id = q.getParentDatabaseId();
    auto& query_gt_traj = groundTruthTrajectories[test_id];
    auto& query_vo_traj = voTrajectories[test_id];
    if (query_gt_traj.find(q.getFrame()->index) == query_gt_traj.end()) {
      query_gt_traj[q.getFrame()->index] =
          make_pair(query_R_gt.clone(), query_t_gt.clone());

      dso::SE3 pose(q.getFrame()->loadPose());
      Mat query_R_vo(3, 3, CV_64F), query_t_vo(3, 1, CV_64F);
      eigen2cv(pose.rotationMatrix(), query_R_vo);
      eigen2cv(pose.translation(), query_t_vo);

      query_vo_traj[q.getFrame()->index] = make_pair(query_R_vo, query_t_vo);
    }

    auto& db_to_q_traj = dbToQueryTrajectories[train_id][test_id];
    db_to_q_traj[q.getFrame()->index] =
        make_tuple(R.clone(), t.clone(), top_result.frame.index, countNonZero(inlier_mask));

    // FIXME
    // For PnP RANSAC, this returns the global pose
    // (For 4pt stereo, this is only a relative pose)
    // Since the following calculations are predicated on the local pose, also
    // estimate the pose of the database frame, and subtract that out
    Mat R_db, t_db, inlier_mask_db;
    internalDoMatching(top_result, database_pts, database_pts, inlier_mask_db,
                       R_db, t_db);
    R = R * R_db.t();
    Mat t_db_inv = -R_db.t() * t_db;
    t += R * t_db;

    // Note: iff x = nan, then x != x is true.
    // Use this to detect nan-filled matrices.
    if ((countNonZero(db_t_gt != db_t_gt) > 0) ||
        countNonZero(query_t_gt != query_t_gt) ||
        (countNonZero(db_R_gt != db_R_gt) > 0) ||
        countNonZero(query_R_gt != query_R_gt)) {
      // For the TUM dataset, poses are only known at the beginning and
      // end, so nans are reasonable. For other datasets, this should not
      // occur. If we wanted to compare to full trajectories, we could
      // compute our own SFM pipeline, although that biases our result.
      queries_without_gt++;
      return;
    }

    // This is only known up to scale. So cheat by fixing the scale to the
    // ground truth.
    if (!needs3dDatabasePoints() && !needs3dQueryPoints()) {
      t *= norm(db_t_gt - query_t_gt) / norm(t);
    } else {
      // Even if we're matching 3D points, the scale of the ground truth
      // and scale of the reconstruction might not be the same. So
      // rescale to address that.
      // TODO: just rescale the reconstruction once at the beginning.
      // TODO: use the reconstruction as the ground truth, so no need for
      // rescaling.
      // TODO: what happens if t is very close to 0? or if db_t_gt ==
      // query_t_gt?
      t *= norm(db_t_gt - query_t_gt) / norm(t);
    }

    if (R.empty() || t.empty()) {
      cout << "invalid transformation! skipping..." << endl;
      return;
    }

    Mat estimated_R = db_R_gt * R;
    Mat estimated_t = db_R_gt * t + db_t_gt;

    double translation_error = norm(query_t_gt, estimated_t);
    // compute angle between rotation matrices
    Mat rotation_diff = (query_R_gt * estimated_R.t());
    // To measure difference, compute the angle when represented in axis-angle
    float angle_error_rad =
        acos((trace(rotation_diff)[0] - 1.0) /
             2.0);
    float angle_error_deg = (180. * angle_error_rad / M_PI);

    if ((translation_error < epsilon_translation) &&
        (angle_error_deg < epsilon_angle_deg)) {
      epsilon_accurate_queries++;
    }

    if (save_first_trajectory && q.getParentDatabaseId() == 0 &&
        top_result.frame.dbId == 0) {
      actualAndExpectedPoses.insert(
          make_pair(q.getFrame()->index, make_pair(estimated_t, query_t_gt)));
    }
  }

  void printResults(int total_queries) {
    double reg_accuracy =
        static_cast<double>(high_inlier_queries) / total_queries;
    double loc_accuracy = static_cast<double>(epsilon_accurate_queries) /
                          (total_queries - queries_without_gt);

    cout << endl;
    cout << "Processed " << total_queries << " queries!" << endl;
    cout << "Registration accuracy (num inliers after pose recovery "
            ">= 12): "
         << reg_accuracy << endl;
    cout << "Localization accuracy (error < " << epsilon_translation << ", "
         << epsilon_angle_deg << " deg): " << loc_accuracy << endl;

    if (save_first_trajectory) {
      ofstream actual_file("actual.txt");
      ofstream expected_file("expected.txt");
      for (const auto& element : actualAndExpectedPoses) {
        const Mat& actual = element.second.first;
        const Mat& expected = element.second.second;
        actual_file << actual.at<float>(0, 0) << " " << actual.at<float>(1, 0)
                    << " " << actual.at<float>(2, 0) << endl;
        expected_file << expected.at<float>(0, 0) << " "
                      << expected.at<float>(1, 0) << " "
                      << expected.at<float>(2, 0) << endl;
      }
      actual_file.close();
      expected_file.close();
    }
  }
  void saveActualAndEstimatedTrajectories(const fs::path& path) {
    fs::path results_dir = path / "results";
    fs::create_directories(results_dir);

    for (auto& element : groundTruthTrajectories) {
      int dbid = element.first;
      auto& trajectory = element.second;

      stringstream ss;
      ss << "gt_" << setfill('0') << setw(2) << dbid << ".txt";
      ofstream ofs((results_dir / ss.str()).string());
      for (auto& frame : trajectory) {
        int frame_id = frame.first;
        Mat R = frame.second.first;
        Mat t = frame.second.second;
        if (R.type() != CV_32F) {
          R.convertTo(R, CV_32F);
        }
        if (t.type() != CV_32F) {
          t.convertTo(t, CV_32F);
        }

        // format on each line:
        // frameid t0 t1 t2 R00 R01 R02 R10 R11 R12 R20 R21 R22
        ofs << frame_id;

        for (int i = 0; i < 3; ++i) {
          ofs << " " << t.at<float>(i);
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            ofs << " " << R.at<float>(i, j);
          }
        }
        ofs << endl;
      }
      ofs.close();
    }

    for (auto& element : voTrajectories) {
      int dbid = element.first;
      auto& trajectory = element.second;

      stringstream ss;
      ss << "vo_" << setfill('0') << setw(2) << dbid << ".txt";
      ofstream ofs((results_dir / ss.str()).string());
      for (auto& frame : trajectory) {
        int frame_id = frame.first;
        Mat R = frame.second.first;
        Mat t = frame.second.second;
        if (R.type() != CV_32F) {
          R.convertTo(R, CV_32F);
        }
        if (t.type() != CV_32F) {
          t.convertTo(t, CV_32F);
        }

        // format on each line:
        // frameid t0 t1 t2 R00 R01 R02 R10 R11 R12 R20 R21 R22
        ofs << frame_id;

        for (int i = 0; i < 3; ++i) {
          ofs << " " << t.at<float>(i);
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            ofs << " " << R.at<float>(i, j);
          }
        }
        ofs << endl;
      }
      ofs.close();
    }

    for (auto& query_trajectories : dbToQueryTrajectories) {
      int train_dbid = query_trajectories.first;
      for (auto& element : query_trajectories.second) {
        int test_dbid = element.first;
        auto& trajectory = element.second;
        stringstream ss;
        ss << "reloc_" << setfill('0') << setw(2) << test_dbid << "_refframe_"
           << setfill('0') << setw(2) << train_dbid << ".txt";
        ofstream ofs((results_dir / ss.str()).string());
        for (auto& frame : trajectory) {
          int test_frame_id = frame.first;
          Mat R = get<0>(frame.second);
          Mat t = get<1>(frame.second);
          int train_frame_id = get<2>(frame.second);
          int inlier_counts = get<3>(frame.second);
          if (R.type() != CV_32F) {
            R.convertTo(R, CV_32F);
          }
          if (t.type() != CV_32F) {
            t.convertTo(t, CV_32F);
          }

          // format
          // frameid refid inlier_count t0 t1 t2 R00 R01 R02 R10 R11 R12 R20 R21 R22
          // note: refid is the frame that was used to estimate the pose of the
          // current frame, but in general it will not have the same pose. It's
          // only provided for completeness.

          ofs << test_frame_id << " " << train_frame_id << " " << inlier_counts;

          for (int i = 0; i < 3; ++i) {
            ofs << " " << t.at<float>(i);
          }
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              ofs << " " << R.at<float>(i, j);
            }
          }
          ofs << endl;
        }
        ofs.close();
      }
    }
  }

  unsigned int epsilon_accurate_queries = 0;
  unsigned int high_inlier_queries = 0;
  unsigned int queries_without_gt = 0;

  float inlierFractionSum = 0;
  int inlierFractionCount = 0;


 protected:
  virtual void internalDoMatching(const Result& top_result,
                                  const vector<Point2f>& query_pts,
                                  const vector<Point2f>& database_pts,
                                  Mat& inlier_mask, Mat& R, Mat& t) = 0;

  Mat K;

 private:
  map<int, pair<Mat, Mat>> actualAndExpectedPoses;

  // dbToQueryTrajectories[d][q][fq] contains a tuple (Mat, Mat, int) of the
  // relocalized pose (rotation and translation Mats, resp.) and fdb (the
  // corresponding frame index in the database), for frame fq from query
  // sequence q in database sequence d's frame of reference
  map<int, map<int, map<int, tuple<Mat, Mat, int, int>>>> dbToQueryTrajectories;
  // groundTruthTrajectories[d][f] contains the ground truth pose (rotation and
  // translation Mats, resp.) of frame f in database sequence d
  map<int, map<int, pair<Mat, Mat>>> groundTruthTrajectories;
  // voTrajectories[d][f] contains the poses (rotation and translation Mats,
  // resp.) estimated from visual odometry (DSO) truth pose of frame f in
  // database sequence d
  map<int, map<int, pair<Mat, Mat>>> voTrajectories;
};

// Match between images and compute homography
class Match_2d_2d_5point : public MatchingMethod {
  bool needs3dDatabasePoints() override { return false; }
  bool needs3dQueryPoints() override { return false; }

  void internalDoMatching(const Result& top_result,
                          const vector<Point2f>& query_pts,
                          const vector<Point2f>& database_pts, Mat& inlier_mask,
                          Mat& R, Mat& t) override {
    int ransac_threshold = 3;  // in pixels, for the sampson error
    // TODO: the number of inliers registered for trainset images is
    // extremely low. Perhaps they are coplanar or otherwise
    // ill-conditioned? Setting confidence to 1 forces RANSAC to use
    // its default maximum number of iterations (1000), but it would
    // be better to filter the data, or increase the max number of
    // trials.
    double confidence = 0.999999;

    // pretty sure this is the correct order
    Mat E = findEssentialMat(database_pts, query_pts, K, RANSAC, confidence,
                             ransac_threshold, inlier_mask);
    recoverPose(E, database_pts, query_pts, K, R, t, inlier_mask);
    // watch out: E, R, & t are double precision (even though we only ever
    // passed floats)

    // Convert from project matrix parameters (world to camera) to camera pose
    // (camera to world)
    R = R.t();
    t = -R * t;
  }
};
// Match 2D to 3D using image retrieval as a proxy for possible co-visibility,
// and use depth values (from slam) to recover transformation
class Match_2d_3d_pnp : public MatchingMethod {
  bool needs3dDatabasePoints() override { return true; }
  bool needs3dQueryPoints() override { return false; }

  void internalDoMatching(const Result& top_result,
                          const vector<Point2f>& query_pts,
                          const vector<Point2f>& database_pts, Mat& inlier_mask,
                          Mat& R, Mat& t) override {
    // lookup scene coords from database_pts
    SceneCoordinateMap scene_coords = top_result.frame.loadSceneCoordinates();
    Mat tmp;
    vector<KeyPoint> keypoints;
    top_result.frame.loadDescriptors(keypoints, tmp);
    for (const auto& kpt : keypoints) {
      const auto& point = kpt.pt;
      // every keypoint should be at a valid depth pixel
      assert(scene_coords.coords.find(make_pair(static_cast<int>(point.y),
                                                static_cast<int>(point.x))) !=
             scene_coords.coords.end());
    }

    vector<Point3f> scene_coord_vec;
    scene_coord_vec.reserve(database_pts.size());
    for (const auto& point : database_pts) {
      assert(scene_coords.coords.find(make_pair(static_cast<int>(point.y),
                                                static_cast<int>(point.x))) !=
             scene_coords.coords.end());
      scene_coord_vec.push_back(scene_coords.coords
                                    .at(make_pair(static_cast<int>(point.y),
                                                  static_cast<int>(point.x)))
                                    .coord);
    }

    if (scene_coord_vec.size() == 0) {
      R = Mat::eye(3, 3, CV_32F);
      t = Mat::zeros(3, 1, CV_32F);
      inlier_mask = Mat::zeros(query_pts.size(), 1, CV_32SC1);
      return;
    }

    Mat rvec, inliers;

//    cout << "\tstarting ransac..." << endl;
    sdl::solvePnPRansac(scene_coord_vec, query_pts, K, rvec, t,
                        ransac_num_iters, ransac_threshold, ransac_confidence,
                        inliers, true);
//    sdl::solvePreemptivePnPRansac(scene_coord_vec, query_pts, K, rvec, t,
//                                  ransac_num_iters, ransac_batch_size,
//                                  ransac_threshold, inliers, true);
    //    cout << "\transac complete!" << endl;
    Rodrigues(rvec, R);

    R.convertTo(R, CV_32F);
    t.convertTo(t, CV_32F);

    // Convert from project matrix parameters (world to camera) to camera pose
    // (camera to world)
    R = R.t();
    t = -R * t;

    inlier_mask = Mat::zeros(query_pts.size(), 1, CV_32SC1);
    for (int i = 0; i < inliers.rows; ++i) {
      inlier_mask.at<int32_t>(inliers.at<int32_t>(i, 0)) = 1;
    }
  }
};
// TODO: direct 3D-3D matching using depth maps? Or 3D-2D matching using a
// prioritized search, as in Sattler et al. 2011?

unique_ptr<MatchingMethod> matching_method;

void usage(char** argv, const po::options_description commandline_args) {
  cout << "Usage: " << argv[0] << " [options]" << endl;
  cout << commandline_args << "\n";
}

void parseArguments(int argc, char** argv) {
  // Arguments, can be specified on commandline or in a file settings.config
  po::options_description commandline_exclusive(
      "Allowed options from terminal");
  commandline_exclusive.add_options()("help", "Print this help message.")(
      "config", po::value<string>(),
      "Path to a config file, which can specify any other argument.");

  po::options_description general_args(
      "Allowed options from terminal or config file");
  general_args.add_options()
      ("scene", po::value<string>(), "Type of scene. Currently the only allowed type is 7scenes, tum, or cambridge.")
      ("datadir", po::value<string>()->default_value(""), "Directory of the scene dataset. For datasets composed"
          " of several scenes, this should be the appropriate subdirectory.")
      ("vocabulary_size", po::value<int>()->default_value(100000), "Size of the visual vocabulary.")
      ("cache", po::value<string>()->default_value(""), "Directory to cache intermediate results, ie"
          " descriptors or visual vocabulary, between runs.")
      ("epsilon_angle_deg", po::value<double>()->default_value(5), "Angle in degrees for a pose to be considered accurate.")
      ("epsilon_translation", po::value<double>()->default_value(0.05), "Distance in the scene coordinate system for a pose"
          " to be considered accurate.")
      ("pose_estimation_method", po::value<string>()->default_value(""), "Robust estimation method to determine pose"
          " after image retrieval. Can be '5point_E' to decompose an essential matrix (and use the ground-truth scale),"
          " 'PNP' to compute a perspective-n-point solution (requires mapping_method to be specified), or left empty.")
      ("use_caffe_descriptor", po::value<bool>()->default_value(false), "Compute a dense descriptor using the provided"
          "caffe model in caffe_prototxt and caffe_caffemodel.")
      ("caffe_prototxt", po::value<string>(), "The prototxt to use to evaulate the descriptor.")
      ("caffe_caffemodel", po::value<string>(), "The trained caffemodel with associated weights. The argument of "
          "caffe_prototxt may have a slightly different structure; only layers with matching names will have their weights "
          "copied over.")
      ("matching_ratio_threshold", po::value<double>()->default_value(0.7), "Threshold for the ratio test. For each "
          "descriptor in a query image, a match is formed if the distance to the first nearest neighbor is less than "
          "this fraction of the distance to the second nearest neighbor.")
      ("seed_ground_truth_poses", po::value<bool>()->default_value(false), "To prevent drift, force the coarse tracker to "
          "use ground truth poses as the initial guess for the motion model.");

  po::options_description commandline_args;
  commandline_args.add(commandline_exclusive).add(general_args);

  // check for config file
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(commandline_exclusive)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    // print help and exit
    usage(argv, commandline_args);
    exit(0);
  }

  if (vm.count("config")) {
    ifstream ifs(vm["config"].as<string>());
    if (!ifs) {
      cout << "could not open config file " << vm["config"].as<string>()
           << endl;
      exit(1);
    }
    vm.clear();
    // since config is added last, commandline args have precedence over
    // args in the config file.
    po::store(
        po::command_line_parser(argc, argv).options(commandline_args).run(),
        vm);
    po::store(po::parse_config_file(ifs, general_args), vm);
  } else {
    vm.clear();
    po::store(
        po::command_line_parser(argc, argv).options(commandline_args).run(),
        vm);
  }
  po::notify(vm);

  fs::path cache_dir = fs::path(vm["cache"].as<string>());

  vocabulary_size = vm["vocabulary_size"].as<int>();
  epsilon_angle_deg = vm["epsilon_angle_deg"].as<double>();
  epsilon_translation = vm["epsilon_translation"].as<double>();
  string matching_method_str = vm["pose_estimation_method"].as<string>();
  if (matching_method_str.length() == 0 ||
      matching_method_str.find("5point_E") == 0) {
    matching_method = unique_ptr<MatchingMethod>(new Match_2d_2d_5point());
  } else if (matching_method_str.find("PNP") == 0 || matching_method_str.find("DLT") == 0) {
    // daniel: "DLT" is an incorrect name, but I don't want to change all my config files
    matching_method = unique_ptr<MatchingMethod>(new Match_2d_3d_pnp());
  } else {
    throw runtime_error("Invalid value for pose_estimation_method!");
  }

  if (vm.count("scene")) {
    string scene_type(vm["scene"].as<string>());
    // currently only one supported scene
    if (scene_type.find("7scenes") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      SevenScenesParser* parser = new SevenScenesParser(directory);
      parser->setCache(cache_dir);
      scene_parser = unique_ptr<SceneParser>(parser);
    } else if (scene_type.find("tum") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      TumParser* parser = new TumParser(directory);
      parser->setCache(cache_dir);
      scene_parser = unique_ptr<SceneParser>(parser);
    } else if (scene_type.find("cambridge") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      CambridgeLandmarksParser* parser =
          new CambridgeLandmarksParser(directory);
      parser->setCache(cache_dir);
      scene_parser = unique_ptr<SceneParser>(parser);
    } else {
      cout << "Invalid value for argument 'scene'." << endl;
      usage(argv, commandline_args);
      exit(1);
    }
  } else {
    cout << "Argument 'scene' is required." << endl;
    usage(argv, commandline_args);
    exit(1);
  }

  use_caffe_descriptor = vm["use_caffe_descriptor"].as<bool>();
  if (use_caffe_descriptor) {
    caffe_prototxt = vm["caffe_prototxt"].as<string>();
    caffe_caffemodel = vm["caffe_caffemodel"].as<string>();
  }

  seed_ground_truth_poses = vm["seed_ground_truth_poses"].as<bool>();

  ratio_threshold = vm["matching_ratio_threshold"].as<double>();
}

}  // namespace sdl

int main(int argc, char** argv) {
  sdl::parseArguments(argc, argv);

  vector<sdl::Database> dbs;
  vector<pair<vector<reference_wrapper<sdl::Database>>, vector<sdl::Query>>>
      dbs_with_queries;

  sdl::scene_parser->parseScene(dbs, dbs_with_queries, sdl::seed_ground_truth_poses);

  Ptr<Feature2D> sift = xfeatures2d::SIFT::create(20000, 3, 0.005, 80);

  Ptr<Feature2D> detector;
  if (sdl::matching_method->needs3dDatabasePoints()) {
    if (sdl::use_caffe_descriptor) {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
      std::unique_ptr<caffe::Net<float>> net(
          sdl::readCaffeModel(sdl::caffe_prototxt, sdl::caffe_caffemodel));
      detector =
          Ptr<Feature2D>(new sdl::DenseDescriptorFromCaffe(move(net)));
    } else {
      detector = Ptr<Feature2D>(new sdl::NearestDescriptorAssigner(*sift));
    }
  } else {
    detector = sift;
  }

  for (sdl::Database& db : dbs) {
    db.setVocabularySize(sdl::vocabulary_size);
    db.setupFeatureDetector(sdl::matching_method->needs3dDatabasePoints());
    db.setDescriptorExtractor(detector);
    db.setBowExtractor(makePtr<BOWSparseImgDescriptorExtractor>(
        sift, FlannBasedMatcher::create()));

    db.runVoAndComputeDescriptors();
  }

  // all query frames should be initialized
  for (auto& train_query : dbs_with_queries) {
    for (sdl::Query& query : train_query.second) {
      if (!query.getFrame()->descriptorsExist()) {
        throw runtime_error("A query does not yet have descriptors!");
      }
    }
  }

  // Free feature detector, no longer needed
  if (sdl::use_caffe_descriptor) {
    detector.staticCast<sdl::DenseDescriptorFromCaffe>()->freeNet();
  }

  for (pair<vector<reference_wrapper<sdl::Database>>, vector<sdl::Query>>&
           train_query : dbs_with_queries) {
    sdl::Database& db = train_query.first[0];
    if (train_query.second.size() == 0) {
      // no queries, ie this is a test set. Don't bother making an index.
      continue;
    }
    db.assignAndBuildIndex(train_query.first);
  }

  int orig_query_count = 0;
  int pruned_query_count = 0;
  for (auto& train_query : dbs_with_queries) {
    auto& queries = train_query.second;
    orig_query_count += queries.size();
    queries.erase(std::remove_if(queries.begin(), queries.end(),
                                 [](sdl::Query& q) {
                                   return q.getFrame()->countDescriptors() == 0;
                                 }),
                  queries.end());
    pruned_query_count += queries.size();
  }
  // Ideally most queries should be valid. :(
  // If pruned_query_count is nonzero, it's probably due to a lack of keyframe
  // density during visual odometry, and discarding empty frames isn't so bad.
  // It might be possible to catch non-keyframes which just have keyframe depth
  // pixels traced on them, but the traces and poses will not be optimized, so
  // the results will be poor.
  cout << "Have " << pruned_query_count << " valid queries after removing "
       << (orig_query_count - pruned_query_count) << " empty queries." << endl;

  // assumes all sequences have same calibration
  Mat K = dbs[0].getCalibration();
  sdl::matching_method->setK(K);

  int num_to_return = 20;

  int total_queries = 0;

  int num_displayed = 0;
  for (auto& train_query : dbs_with_queries) {
    for (sdl::Query& query : train_query.second) {
      total_queries++;

      unsigned int total_so_far = total_queries;
      unsigned int total = pruned_query_count;
      if (total_so_far % 10 == 0 || total_so_far == total) {
        cout << "\r" << fixed << setprecision(4)
             << static_cast<double>(total_so_far) / total * 100. << "% ("
             << total_so_far << "/" << total << ")" << flush;
      }

      // All of these should be backed by the same inverted index, so just pick
      // the first
      sdl::Database& db = train_query.first[0];
      vector<reference_wrapper<sdl::Database>> all_training_dbs =
          train_query.first;

      cout << "\tquerying index..." << endl;
      vector<sdl::Result> results =
          db.lookup(query, num_to_return, all_training_dbs);
      cout << "\tquery complete!" << endl;
      unsigned int top_index = sdl::use_second_best_for_debugging ? 1 : 0;
      if (results.size() <= top_index) {
        cout << results.size() << " results returned!" << endl;
        continue;
      }

      Mat q_descriptors;
      vector<KeyPoint> query_keypoints;
      query.getFrame()->loadDescriptors(query_keypoints, q_descriptors);

      Ptr<DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
//      Ptr<DescriptorMatcher> matcher = cv::BFMatcher::create();
      vector<int> num_inliers(results.size(), 0);
      if (!sdl::use_second_best_for_debugging) {
        // rerank by num inliers after geometric verification
        for (unsigned int result_idx = 0; result_idx < results.size(); ++result_idx) {
          vector<KeyPoint> db_keypoints;
          Mat db_descriptors;
          results[result_idx].frame.loadDescriptors(db_keypoints,
                                                    db_descriptors);

          // check ratio test
          map<int, list<DMatch>> best_match_per_trainidx(sdl::doRatioTest(
              q_descriptors, db_descriptors, query_keypoints, db_keypoints,
              matcher, sdl::ratio_threshold, sdl::use_caffe_descriptor));
          vector<Point2f> p1, p2;
          p1.reserve(best_match_per_trainidx.size());
          p2.reserve(best_match_per_trainidx.size());
          for (auto& element : best_match_per_trainidx) {
            for (auto& match : element.second) {
              p1.push_back(query_keypoints[match.queryIdx].pt);
              p2.push_back(db_keypoints[match.trainIdx].pt);
            }
          }

          if (p1.size() == 0) {
            continue;
          }

          if (sdl::use_5pt_for_verif) {
            double prob = 0.999;
            double threshold = 1.0;
            Mat inlier_mask;
            findEssentialMat(p1, p2, K, cv::RANSAC, prob, threshold,
                             inlier_mask);
            num_inliers[result_idx] = countNonZero(inlier_mask);
          } else {
            sdl::SceneCoordinateMap scene_coords =
                results[result_idx].frame.loadSceneCoordinates();
            vector<KeyPoint> keypoints;

            vector<Point3f> scene_coord_vec;
            scene_coord_vec.reserve(p2.size());
            for (const auto& point : p2) {
              assert(
                  scene_coords.coords.find(make_pair(
                      static_cast<int>(point.y), static_cast<int>(point.x))) !=
                  scene_coords.coords.end());
              scene_coord_vec.push_back(
                  scene_coords.coords
                      .at(make_pair(static_cast<int>(point.y),
                                    static_cast<int>(point.x)))
                      .coord);
            }

            Mat rvec, t, inliers;

            //            cout << "\tstarting ransac for geom verif..." << endl;

                        sdl::solvePnPRansac(scene_coord_vec, p1, K, rvec, t,
                                            sdl::ransac_num_iters,
                                            sdl::ransac_threshold,
                                            sdl::ransac_confidence, inliers,
                                            false);
//            sdl::solvePreemptivePnPRansac(
//                scene_coord_vec, p1, K, rvec, t, sdl::ransac_num_iters,
//                sdl::ransac_batch_size, sdl::ransac_threshold, inliers, false);

            //            cout << "\transac complete!" << endl;
            num_inliers[result_idx] = inliers.rows * inliers.cols;
          }
        }

        top_index =
            distance(num_inliers.begin(),
                     max_element(num_inliers.begin(), num_inliers.end()));
      }
      sdl::Result& top_result = results[top_index];

      // for now, just the first image
      vector<Point2f> query_pts;
      vector<Point2f> database_pts;

      vector<KeyPoint> db_keypoints;
      Mat db_descriptors;
      top_result.frame.loadDescriptors(db_keypoints, db_descriptors);

      // ignore the matches used during inverted index voting. just compute our
      // own.
      // check ratio test
      map<int, list<DMatch>> best_match_per_trainidx(sdl::doRatioTest(
          q_descriptors, db_descriptors, query_keypoints, db_keypoints, matcher,
          sdl::ratio_threshold, sdl::use_caffe_descriptor));
      vector<DMatch> matches;
      matches.reserve(best_match_per_trainidx.size());
      for (auto& element : best_match_per_trainidx) {
        for (auto& match : element.second) {
          matches.push_back(match);
        }
      }

      // display the result in a pretty window
      if (num_displayed < 20 && sdl::display_top_matching_images &&
          static_cast<unsigned int>(top_result.frame.dbId) !=
              query.getParentDatabaseId() &&
          query.getFrame()->index > 200) {
        num_displayed++;
        int width = 640, height = 640;
        int rows = 3, cols = 3;
        int bordersize = 3;
        Mat grid(height, width, CV_8UC3);
        rectangle(grid, Point(0, 0), Point(width, height),
                  Scalar(255, 255, 255), FILLED, LINE_8);
        rectangle(grid, Point(0, 0), Point(width / cols, height / rows),
                  Scalar(0, 0, 255), FILLED, LINE_8);

        for (int c = 0; c < cols; ++c) {
          for (int r = 0; r < rows; ++r) {
            Point start(bordersize + c * width / cols,
                        bordersize + r * height / rows);
            Point end((c + 1) * width / cols - bordersize,
                      (r + 1) * height / rows - bordersize);
            int tilewidth = end.x - start.x;
            int tileheight = end.y - start.y;

            Mat image;
            string text;
            if (r == 0 && c == 0) {
              // load query
              image = query.readColorImage();
              stringstream ss;
              ss << "query, db=" << query.getParentDatabaseId();
              text = ss.str();

              // Draw all the possible places for matches
              vector<KeyPoint> query_keypoints;
              Mat query_descriptors;
              query.getFrame()->loadDescriptors(query_keypoints,
                                                query_descriptors);
              for (auto& element : query_keypoints) {
                // draw larger than 1 pixel, so it shows up after resizing
                circle(image, element.pt, 3, Scalar(0, 0, 255), -1);
              }
            } else {
              // load db match
              int index = cols * r + c - 1;
              image = results[index].frame.imageLoader();
              stringstream ss;
              ss << "db=" << results[index].frame.dbId << ", rank=" << index
                 << ", #inl=" << num_inliers[index];
              text = ss.str();

              // also plot locations of putative matches
              vector<KeyPoint> db_keypoints;
              Mat db_descriptors;
              results[index].frame.loadDescriptors(db_keypoints,
                                                   db_descriptors);
              // check ratio test
              map<int, list<DMatch>> best_match_per_trainidx(sdl::doRatioTest(
                  q_descriptors, db_descriptors, query_keypoints, db_keypoints,
                  matcher, sdl::ratio_threshold, sdl::use_caffe_descriptor));
              for (auto& element : best_match_per_trainidx) {
                for (auto& match : element.second) {
                  // draw larger than 1 pixel, so it shows up after resizing
                  circle(image, db_keypoints[match.trainIdx].pt, 3,
                         Scalar(0, 0, 255), -1);
                }
              }
            }


            resize(image, image, Size(tilewidth, tileheight));
            putText(image, text, Point(10, 15),
                    FONT_HERSHEY_PLAIN, 0.9, Scalar(255, 255, 255));
            image.copyTo(grid(Rect(start.x, start.y, tilewidth, tileheight)));
          }
        }
        string window_name("Closest Images");
        namedWindow(window_name, WINDOW_AUTOSIZE);
        imshow(window_name, grid);
        waitKey(0);
        destroyWindow(window_name);

        {
          Mat query_img = query.getFrame()->imageLoader();
          Mat result = top_result.frame.imageLoader();
          Mat stereo;
          stereo.create(query_img.rows, query_img.cols + result.cols,
                        query_img.type());

          query_img.copyTo(stereo.colRange(0, query_img.cols));
          result.copyTo(
              stereo.colRange(query_img.cols, query_img.cols + result.cols));

          vector<cv::KeyPoint> keypoints_q;
          Mat descriptors_q;
          query.getFrame()->loadDescriptors(keypoints_q, descriptors_q);
          vector<cv::KeyPoint> keypoints_db;
          Mat descriptors_db;
          top_result.frame.loadDescriptors(keypoints_db, descriptors_db);
          drawKeypoints(stereo.colRange(0, query_img.cols), keypoints_q,
                        stereo.colRange(0, query_img.cols), Scalar::all(-1),
                        DrawMatchesFlags::DEFAULT);
          drawKeypoints(stereo.colRange(query_img.cols, 2 * query_img.cols),
                        keypoints_db,
                        stereo.colRange(query_img.cols, 2 * query_img.cols),
                        Scalar::all(-1), DrawMatchesFlags::DEFAULT);

          string window_name("Keypoints");
          namedWindow(window_name, WINDOW_AUTOSIZE);
          imshow(window_name, stereo);
          waitKey(0);
          destroyWindow(window_name);
        }

        {
          // show correspondences between query and best match
          Mat query_img = query.getFrame()->imageLoader();
          Mat result = top_result.frame.imageLoader();
          Mat stereo;
          stereo.create(query_img.rows, query_img.cols + result.cols, query_img.type());

          query_img.copyTo(stereo.colRange(0, query_img.cols));
          result.copyTo(stereo.colRange(query_img.cols, query_img.cols + result.cols));

          const vector<KeyPoint>& query_keypoints = query.getKeypoints();
          vector<KeyPoint> result_keypoints;
          Mat dummy;
          top_result.frame.loadDescriptors(result_keypoints, dummy);

          for (auto& correspondence : matches) {
            Point2f from = query_keypoints[correspondence.queryIdx].pt;
            Point2f to = result_keypoints[correspondence.trainIdx].pt +
                         Point2f(query_img.cols, 0);

            circle(stereo, from, 3, Scalar(0, 0, 255));
            circle(stereo, to, 3, Scalar(0, 0, 255));
            line(stereo, from, to, Scalar(255, 0, 0));
          }

          string window_name("Correspondences in top result");
          namedWindow(window_name, WINDOW_AUTOSIZE);
          imshow(window_name, stereo);
          waitKey(0);
          destroyWindow(window_name);
        }

        {
          // visualize descriptor distance
          Mat query_img = query.getFrame()->imageLoader();
          Mat result = top_result.frame.imageLoader();
          Mat stereo;
          stereo.create(query_img.rows, query_img.cols + result.cols, query_img.type());

          query_img.copyTo(stereo.colRange(0, query_img.cols));
          result.copyTo(stereo.colRange(query_img.cols, query_img.cols + result.cols));

          vector<KeyPoint> query_keypoints;
          Mat query_descriptors;
          query.getFrame()->loadDescriptors(query_keypoints, query_descriptors);

          // just choose a keypoint in the middle
          int mid = query_keypoints.size() / 2;
          KeyPoint& kpt = query_keypoints[mid];
          circle(stereo, kpt.pt, 5, cv::Scalar(0, 0, 0), -1);
          circle(stereo, kpt.pt, 4, cv::Scalar(0, 0, 255), -1);
          Mat descr = query_descriptors.row(mid);

          vector<KeyPoint> result_keypoints;
          Mat result_descriptors;
          top_result.frame.loadDescriptors(result_keypoints,
                                           result_descriptors);

          Mat kpts_as_distance(1, result_keypoints.size() + 1, CV_32F);
          float maxdist = 0;
          for (unsigned int i = 0; i < result_keypoints.size(); ++i) {
            float dist = cv::norm(result_descriptors.row(i) - descr);
            maxdist = max(maxdist, dist);
            kpts_as_distance.at<float>(0, i) = dist;
          }
          kpts_as_distance.at<float>(0, result_keypoints.size()) = 0.0f;
          kpts_as_distance = (1 - kpts_as_distance / maxdist) * 255;
          kpts_as_distance.convertTo(kpts_as_distance, CV_8UC1);

          Mat kpts_color;
          cv::applyColorMap(kpts_as_distance, kpts_color,
                            cv::ColormapTypes::COLORMAP_JET);
          for (unsigned int i = 0; i < result_keypoints.size(); ++i) {
            cv::Vec3b color = kpts_color.at<cv::Vec3b>(0, i);
            circle(stereo,
                   result_keypoints[i].pt + cv::Point2f(query_img.cols, 0), 2,
                   color, -1);
          }

          string window_name("Similarity of keypoints to an arbitrary point");
          namedWindow(window_name, WINDOW_AUTOSIZE);
          imshow(window_name, stereo);
          waitKey(0);
          destroyWindow(window_name);
        }
      }


      //      for (auto& correspondence : top_result.matches) {
      for (auto& correspondence : matches) {
        Mat query_descr = q_descriptors.row(correspondence.queryIdx);
        Mat db_descr = db_descriptors.row(correspondence.trainIdx);

//        double min_dist = cv::norm(query_descr - db_descr);
//        for (int k = 0; k < db_descr.rows; ++k) {
//          min_dist =
//              std::min(min_dist, cv::norm(query_descr - db_descriptors.row(k)));
//        }
//        cout << "(" << correspondence.queryIdx << ", "
//             << correspondence.trainIdx
//             << ", dist=" << cv::norm(query_descr - db_descr) << ", closest dist=" << min_dist << "), ";


        query_pts.push_back(query_keypoints[correspondence.queryIdx].pt);
        database_pts.push_back(db_keypoints[correspondence.trainIdx].pt);
      }
//      cout << endl;

      sdl::matching_method->doMatching(
          query, top_result, query_pts,
          database_pts);
    }
  }

  sdl::matching_method->printResults(total_queries);
  sdl::matching_method->saveActualAndEstimatedTrajectories(
      sdl::scene_parser->getCache());

  return 0;
}
