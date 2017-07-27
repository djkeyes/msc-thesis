#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/operations.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "Datasets.h"
#include "FusedFeatureDescriptors.h"
#include "Relocalization.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

namespace sdl {

bool use_second_best_for_debugging = false;
bool only_first_db_for_debugging = false;
bool display_top_matching_images = false;
bool save_first_trajectory = true;
bool display_stereo_correspondences = false;

int vocabulary_size;
double epsilon_angle_deg;
double epsilon_translation;
bool reuse_first_vocabulary;

unique_ptr<SceneParser> scene_parser;

class MatchingMethod {
 public:
  virtual ~MatchingMethod() = default;

  virtual bool needs3dDatabasePoints() = 0;
  virtual bool needs3dQueryPoints() = 0;

  void doMatching(const Query& q, const Result& top_result, bool same_database,
                  const vector<Point2f>& query_pts,
                  const vector<Point2f>& database_pts) {
    Mat R, t, inlier_mask;
    internalDoMatching(top_result, query_pts, database_pts, inlier_mask, R, t);

    updateResults(q, top_result, same_database, inlier_mask, R, t, query_pts,
                  database_pts);
  }

  void setK(Mat K_) { K = K_; }

  void updateResults(const Query& q, const Result& top_result,
                     bool same_database, Mat inlier_mask, Mat R, Mat t,
                     const vector<Point2f>& query_pts,
                     const vector<Point2f>& database_pts) {
    static int num_displayed = 0;
    if (display_stereo_correspondences && num_displayed < 5) {
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
          color = Scalar(0, 0, 255);
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

      ++num_displayed;
    }

    // should this threshold be a function of the ransac model DOF?
    if (countNonZero(inlier_mask) >= 12) {
      if (same_database) {
        high_inlier_train_queries++;
      } else {
        high_inlier_test_queries++;
      }
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
        make_tuple(R.clone(), t.clone(), top_result.frame.index);

    // FIXME
    // For PnP RANSAC, this returns the global pose
    // (For 4pt stereo, this is only a relative pose)
    // Since the following calculations are predicated on the local pose, also
    // estimate the pose of the database frame, and subtract that out
    Mat R_db, t_db, inlier_mask_db;
    internalDoMatching(top_result, database_pts, database_pts, inlier_mask,
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
      if (same_database) {
        train_queries_without_gt++;
      } else {
        test_queries_without_gt++;
      }
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
      if (same_database) {
        epsilon_accurate_train_queries++;
      } else {
        epsilon_accurate_test_queries++;
      }
    }

    if (save_first_trajectory && q.getParentDatabaseId() == 0 &&
        top_result.frame.dbId == 0) {
      actualAndExpectedPoses.insert(
          make_pair(q.getFrame()->index, make_pair(estimated_t, query_t_gt)));
    }
  }

  void printResults(int total_train_queries, int total_test_queries) {
    double train_reg_accuracy =
        static_cast<double>(high_inlier_train_queries) / total_train_queries;
    double test_reg_accuracy =
        static_cast<double>(high_inlier_test_queries) / total_test_queries;
    double train_loc_accuracy =
        static_cast<double>(epsilon_accurate_train_queries) /
        (total_train_queries - train_queries_without_gt);
    double test_loc_accuracy =
        static_cast<double>(epsilon_accurate_test_queries) /
        (total_test_queries - test_queries_without_gt);

    cout << endl;
    cout << "Processed " << (total_test_queries + total_train_queries)
         << " queries! (" << total_train_queries
         << " queries on their own train set database, and "
         << total_test_queries << " on separate test set databases)" << endl;
    cout << "Train set registration accuracy (num inliers after pose recovery "
            ">= 12): "
         << train_reg_accuracy << endl;
    cout << "Test set registration accuracy (num inliers after pose recovery "
            ">= 12): "
         << test_reg_accuracy << endl;
    cout << "Train set localization accuracy (error < " << epsilon_translation
         << ", " << epsilon_angle_deg << " deg): " << train_loc_accuracy
         << endl;
    cout << "Test set localization accuracy (error < " << epsilon_translation
         << ", " << epsilon_angle_deg << " deg): " << test_loc_accuracy << endl;

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
          if (R.type() != CV_32F) {
            R.convertTo(R, CV_32F);
          }
          if (t.type() != CV_32F) {
            t.convertTo(t, CV_32F);
          }

          // format
          // frameid refid t0 t1 t2 R00 R01 R02 R10 R11 R12 R20 R21 R22
          // note: refid is the frame that was used to estimate the pose of the
          // current frame, but in general it will not have the same pose. It's
          // only provided for completeness.

          ofs << test_frame_id << " " << train_frame_id;

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

  unsigned int epsilon_accurate_test_queries = 0;
  unsigned int high_inlier_test_queries = 0;
  unsigned int epsilon_accurate_train_queries = 0;
  unsigned int high_inlier_train_queries = 0;
  unsigned int test_queries_without_gt = 0;
  unsigned int train_queries_without_gt = 0;

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
  map<int, map<int, map<int, tuple<Mat, Mat, int>>>> dbToQueryTrajectories;
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
class Match_2d_3d_dlt : public MatchingMethod {
  bool needs3dDatabasePoints() override { return true; }
  bool needs3dQueryPoints() override { return false; }

  void internalDoMatching(const Result& top_result,
                          const vector<Point2f>& query_pts,
                          const vector<Point2f>& database_pts, Mat& inlier_mask,
                          Mat& R, Mat& t) override {
    int num_iters = 1000;
    double confidence = 0.9999999;
    double ransac_threshold = 8.0;

    // lookup scene coords from database_pts
    SparseMat scene_coords = top_result.frame.loadSceneCoordinates();
    vector<Point3f> scene_coord_vec;
    scene_coord_vec.reserve(database_pts.size());
    for (const auto& point : database_pts) {
      scene_coord_vec.push_back(scene_coords.value<Point3f>(
          static_cast<int>(point.y), static_cast<int>(point.x)));
    }

    Mat rvec, inliers;

    solvePnPRansac(scene_coord_vec, query_pts, K, noArray(), rvec, t, false,
                   num_iters, ransac_threshold, confidence, inliers);
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
      ("scene", po::value<string>(), "Type of scene. Currently the only allowed type is 7scenes.")
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
          " 'DLT' to compute a direct linear transform (requires mapping_method to be specified), or left empty.")
      ("reuse_first_vocabulary", po::value<bool>()->default_value(false), "Compute a visual vocabulary once by clustering "
          "descriptors in the first database, then re-use it in all subsequent databases. Faster, but potentially much less "
          "accurate.");

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

  fs::path cache_dir;
  if (vm.count("cache")) {
    cache_dir = fs::path(vm["cache"].as<string>());
  }

  vocabulary_size = vm["vocabulary_size"].as<int>();
  epsilon_angle_deg = vm["epsilon_angle_deg"].as<double>();
  epsilon_translation = vm["epsilon_translation"].as<double>();
  reuse_first_vocabulary = vm["reuse_first_vocabulary"].as<bool>();
  string matching_method_str = vm["pose_estimation_method"].as<string>();
  if (matching_method_str.length() == 0 ||
      matching_method_str.find("5point_E") == 0) {
    matching_method = unique_ptr<MatchingMethod>(new Match_2d_2d_5point());
  } else if (matching_method_str.find("DLT") == 0) {
    matching_method = unique_ptr<MatchingMethod>(new Match_2d_3d_dlt());
  } else {
    throw runtime_error("Invalid value for pose_estimation_method!");
  }

  if (vm.count("scene")) {
    string scene_type(vm["scene"].as<string>());
    // currently only one supported scene
    if (scene_type.find("7scenes") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      SevenScenesParser* parser =
          new SevenScenesParser(directory);
      if (!cache_dir.empty()) {
        parser->setCache(cache_dir);
      }
      scene_parser = unique_ptr<SceneParser>(parser);
    } else if (scene_type.find("tum") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      TumParser* parser = new TumParser(directory);
      if (!cache_dir.empty()) {
        parser->setCache(cache_dir);
      }
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
}

}  // namespace sdl

int main(int argc, char** argv) {
  sdl::parseArguments(argc, argv);

  vector<sdl::Database> dbs;
  vector<sdl::Query> queries;

  sdl::scene_parser->parseScene(dbs, queries);

  Ptr<Feature2D> sift = xfeatures2d::SIFT::create(20000, 3, 0.005, 80);

  Ptr<Feature2D> detector;
  if (sdl::matching_method->needs3dDatabasePoints()) {
    detector = Ptr<Feature2D>(new sdl::NearestDescriptorAssigner(*sift));
  } else {
    detector = sift;
  }

  if (sdl::only_first_db_for_debugging) {
    // remove all but the first database.
    dbs.resize(1);
    queries.erase(std::remove_if(queries.begin(), queries.end(),
                                 [&dbs](sdl::Query& q) {
                                   return q.getParentDatabaseId() !=
                                          dbs[0].db_id;
                                 }),
                  queries.end());
  }

  sdl::Database* db_with_vocab(nullptr);
  for (sdl::Database& db : dbs) {
    if (db.hasCachedVocabularyFile()) {
      db_with_vocab = &db;
      break;
    }
  }

  for (sdl::Database& db : dbs) {
    db.setVocabularySize(sdl::vocabulary_size);
    db.setupFeatureDetector(sdl::matching_method->needs3dDatabasePoints());
    db.setDescriptorExtractor(detector);
    db.setBowExtractor(makePtr<BOWSparseImgDescriptorExtractor>(
        sift, FlannBasedMatcher::create()));

    if (sdl::reuse_first_vocabulary && db_with_vocab != nullptr &&
        !db.hasCachedVocabularyFile()) {
      db.copyVocabularyFileFrom(*db_with_vocab);
    }
    db.train();
    if (db_with_vocab == nullptr) {
      db_with_vocab = &db;
    }
  }
  for (sdl::Query& query : queries) {
    // TODO: incorporate previous frames into each query (or maybe just use
    // the cached depth map, which would be fine for cross-database queries)
    query.setupFeatureDetector(sdl::matching_method->needs3dQueryPoints());
    query.setDescriptorExtractor(sift);
    if (!query.getFrame()->descriptorsExist()) {
      query.computeDescriptors();
    }
  }

  int orig_query_count = queries.size();
  queries.erase(std::remove_if(queries.begin(), queries.end(),
                               [](sdl::Query& q) {
                                 return q.getFrame()->countDescriptors() == 0;
                               }),
                queries.end());
  // Ideally most queries should be valid. :(
  // If this is nonzero, it's probably due to a lack of keyframe density
  // during visual odometry, and discarding empty frames isn't so bad.
  int pruned_query_count = queries.size();
  cout << "Have " << pruned_query_count << " valid queries after removing "
       << (orig_query_count - pruned_query_count) << " empty queries." << endl;

  // assumes all sequences have same calibration
  Mat K = dbs[0].getCalibration();
  sdl::matching_method->setK(K);

  int num_to_return = 8;

  unsigned int total_test_queries = 0;
  unsigned int total_train_queries = 0;

  for (unsigned int i = 0; i < dbs.size(); i++) {
    for (unsigned int j = 0; j < queries.size(); j++) {
      if (queries[j].getParentDatabaseId() == dbs[i].db_id) {
        total_train_queries++;
      } else {
        total_test_queries++;
      }
      unsigned int total_so_far = total_test_queries + total_train_queries;
      unsigned int total = dbs.size() * queries.size();
      if (total_so_far % 10 == 0 || total_so_far == total) {
        cout << "\r" << fixed << setprecision(4)
             << static_cast<double>(total_so_far) / total * 100. << "% ("
             << total_so_far << "/" << total << ")" << flush;
      }

      vector<sdl::Result> results = dbs[i].lookup(queries[j], num_to_return);
      unsigned int top_index = sdl::use_second_best_for_debugging ? 1 : 0;
      if (results.size() <= top_index) {
        cout << results.size() << " results returned!" << endl;
        continue;
      }
      sdl::Result& top_result = results[top_index];

      // display the result in a pretty window
      if (j < 5 && sdl::display_top_matching_images) {
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
              image = queries[j].readColorImage();
              text = "query";
            } else {
              // load db match
              int index = cols * r + c - 1;
              image = results[index].frame.imageLoader();
              stringstream ss;
              ss << "rank=" << index;
              text = ss.str();
            }
            resize(image, image, Size(tilewidth, tileheight));
            putText(image, text, Point(tilewidth / 2 - 30, 15),
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
          // show correspondences between query and best match
          int width = 1280, height = 640;
          Mat img(height, width, CV_8UC3);

          Mat query_image = queries[j].readColorImage();
          resize(query_image, query_image, Size(width / 2, height));
          query_image.copyTo(img(Rect(0, 0, width / 2, height)));

          Mat db_image = top_result.frame.imageLoader();
          resize(db_image, db_image, Size(width / 2, height));
          db_image.copyTo(img(Rect(width / 2, 0, width / 2, height)));

          const vector<KeyPoint>& query_keypoints = queries[j].getKeypoints();
          vector<KeyPoint> result_keypoints;
          Mat dummy;
          top_result.frame.loadDescriptors(result_keypoints, dummy);

          for (auto& correspondence : top_result.matches) {
            Point2f from = query_keypoints[correspondence.queryIdx].pt;
            Point2f to = result_keypoints[correspondence.trainIdx].pt +
                         Point2f(width / 2, 0);

            circle(img, from, 3, Scalar(0, 0, 255));
            circle(img, to, 3, Scalar(0, 0, 255));
            line(img, from, to, Scalar(255, 0, 0));
          }

          string window_name("Correspondences in top result");
          namedWindow(window_name, WINDOW_AUTOSIZE);
          imshow(window_name, img);
          waitKey(0);
          destroyWindow(window_name);
        }
      }

      // not enough correspondences to estimate
      if (top_result.matches.size() <= 5) {
        continue;
      }
      // for now, just the first image
      vector<Point2f> query_pts;
      vector<Point2f> database_pts;

      const vector<KeyPoint>& query_keypoints = queries[j].getKeypoints();
      vector<KeyPoint> result_keypoints;
      Mat dummy;
      top_result.frame.loadDescriptors(result_keypoints, dummy);

      for (auto& correspondence : top_result.matches) {
        query_pts.push_back(query_keypoints[correspondence.queryIdx].pt);
        database_pts.push_back(result_keypoints[correspondence.trainIdx].pt);
      }

      sdl::matching_method->doMatching(
          queries[j], top_result,
          queries[j].getParentDatabaseId() == dbs[i].db_id, query_pts,
          database_pts);
    }
  }

  sdl::matching_method->printResults(total_train_queries, total_test_queries);
  sdl::matching_method->saveActualAndEstimatedTrajectories(
      sdl::scene_parser->getCache());

  return 0;
}
