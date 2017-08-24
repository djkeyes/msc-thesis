
#include <algorithm>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "boost/filesystem.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Datasets.h"
#include "Relocalization.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

namespace sdl {

void read_float_or_nan_or_inf(istream& in, float& value) {
  // This isn't especially performant, but works in a pinch
  string tmp;
  in >> tmp;
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
  if (tmp.find("nan") == 0) {
    value = numeric_limits<float>::quiet_NaN();
  } else {
    stringstream ss(tmp);
    ss >> value;
  }
}

set<string> CambridgeLandmarksParser::accumulateSubdirsFromFile(
    const string& filename) {
  if (!fs::exists(filename)) {
    stringstream ss;
    ss << "Test/train file does not exist: " << filename;
    throw runtime_error(ss.str());
  }

  ifstream file_list(filename);
  string line;
  // first 3 lines are fluff
  getline(file_list, line);
  getline(file_list, line);
  getline(file_list, line);

  set<string> result;
  while (getline(file_list, line)) {
    stringstream ss;
    ss << line;
    string rel_path;
    ss >> rel_path;
    result.insert(rel_path.substr(0, rel_path.find('/')));
  }
  return result;
}

int id_from_filename_and_dir(const string& name, const string& dir,
                             bool is_street) {
  if (!is_street) {
    // these files are in the format frameXXXXX.jpg, but 1-indexed, so
    // subtract 1.
    return stoi(name.substr(5, 5)) - 1;
  }

  if (dir == "img") {
    string prefix("image");
    int video = stoi(name.substr(prefix.length(), 1));
    int id = stoi(name.substr(prefix.length() + 2, 6)) - 1;
    if (video == 2) {
      // this many images start with 'image1_', then the counter restarts
      id += 436;
    }
    return id;
  } else if (dir == "img_north") {
    string prefix("image_north_");
    int video = stoi(name.substr(prefix.length(), 1));
    int id = stoi(name.substr(prefix.length() + 2, 4)) - 1;
    if (video == 2) {
      // this many images start with 'image_north_1_', then the counter restarts
      id += 622;
    }
    return id;
  } else if (dir == "img_east") {
    string prefix("image_east_");
    int id = stoi(name.substr(prefix.length(), 4)) - 1;
    return id;
  } else if (dir == "img_south") {
    // there's no image_south_2_. go figure.
    string prefix("image_south_1_");
    int id = stoi(name.substr(prefix.length(), 4)) - 1;
    return id;
  } else if (dir == "img_west") {
    string prefix("image_west_");
    int id = stoi(name.substr(prefix.length(), 4)) - 1;
    return id;
  } else {
    stringstream ss;
    ss << "Unexpected sequence name in Street scene: " << dir;
    throw runtime_error(ss.str());
  }
}

void CambridgeLandmarksParser::parseScene(
    vector<Database>& dbs,
    vector<pair<vector<reference_wrapper<Database>>, vector<Query>>>&
        dbs_with_queries,
    bool inject_ground_truth) {
  // The names in this dataset sometimes follow weird patterns, so load the
  // train and test lists, which give correct directory names

  string test_filename((directory / "dataset_test.txt").string());
  string train_filename((directory / "dataset_train.txt").string());

  // street violates all the naming conventions. come on, guys.
  bool is_street =
      (directory.filename().string().find("Street") != string::npos);

  set<string> test_subdirs(accumulateSubdirsFromFile(test_filename));
  set<string> train_subdirs(accumulateSubdirsFromFile(train_filename));

  // should be disjoint
  for (const string& s : test_subdirs) {
    assert(train_subdirs.find(s) == train_subdirs.end());
  }

  vector<string> all_subdirs;
  all_subdirs.insert(all_subdirs.end(), test_subdirs.begin(),
                     test_subdirs.end());
  all_subdirs.insert(all_subdirs.end(), train_subdirs.begin(),
                     train_subdirs.end());

  sort(all_subdirs.begin(), all_subdirs.end());

  // create empty vecs to store the true training set and all the queries (we
  // can split train and test queries later)
  dbs_with_queries.emplace_back();

  for (int i = all_subdirs.size() - 1; i >= 0; --i) {
    if (!fs::exists(directory / all_subdirs[i])) {
      cout << "Subdirectory " << all_subdirs[i]
           << " doesnt exist! Skipping it..." << endl;
      all_subdirs.erase(all_subdirs.begin() + i);
    }
  }

  for (const string& subdir : all_subdirs) {
    fs::path sequence_dir = directory / subdir;
    dbs.emplace_back();

    Database& cur_db = dbs.back();
    cur_db.db_id = dbs.size() - 1;
    cur_db.setCachePath(cache / subdir);
  }

  for (unsigned int seq_idx = 0; seq_idx < all_subdirs.size(); ++seq_idx) {
    const fs::path& sequence_dir = directory / all_subdirs[seq_idx];
    Database& cur_db = dbs[seq_idx];
    dbsByName.emplace(all_subdirs[seq_idx], dbs[seq_idx]);

    bool in_train_set =
        (train_subdirs.find(all_subdirs[seq_idx]) != train_subdirs.end());
    if (in_train_set) {
      // store all the training databases in the first entry
      dbs_with_queries[0].first.push_back(cur_db);
    } else {
      // just create a pair with zero queries
      dbs_with_queries.emplace_back();
      dbs_with_queries.back().first.push_back(cur_db);
    }

    shared_ptr<ImageFolderReader> reader(new ImageFolderReader(
        sequence_dir.string(), (directory / "colmap-calib.txt").string(), "",
        ""));

    vector<string> sorted_images;
    for (auto file : fs::recursive_directory_iterator(sequence_dir)) {
      string name(file.path().filename().string());
      // note: resampled ends in jpg. originals are png.
      if (name.find(".jpg") == string::npos &&
          name.find(".png") == string::npos) {
        continue;
      }

      int id = id_from_filename_and_dir(name, all_subdirs[seq_idx], is_street);
      unique_ptr<Frame> frame(new sdl::Frame(id, cur_db.db_id));
      frame->setImageLoader([reader, id]() {
        ImageAndExposure* image = reader->getImage(id);
        Mat shallow(image->h, image->w, CV_32FC1, image->image);
        Mat converted;
        shallow.convertTo(converted, CV_8UC1);
        cvtColor(converted, converted, ColorConversionCodes::COLOR_GRAY2RGB);
        delete image;
        return converted;
      });
      frame->setPath(sequence_dir);
      string image_filename(file.path().string());
      frame->setCachePath(cache / sequence_dir.filename());
      sorted_images.push_back(image_filename);

      // store all the queries in the first entry
      // remove this conditional if you want to evaulate the train set on itself
      if (!in_train_set) {
        dbs_with_queries[0].second.emplace_back(cur_db.db_id, frame.get());
      }
      cur_db.addFrame(move(frame));
    }

    sort(sorted_images.begin(), sorted_images.end());

    // use calib from colmap
    cur_db.setMapper(new DsoMapGenerator(
        sequence_dir.string(), (directory / "colmap-calib.txt").string(),
        !in_train_set));
    if (inject_ground_truth) {
      cur_db.setGtPoseLoader([this](const Frame& frame, Mat& R, Mat& t) {
        return loadGroundTruthPose(frame, R, t);
      });
    }
  }

  // preload the ground truth poses, so we don't have to seek every time
  // file is directory/groundtruthSync.txt
  fs::path test_pose_path = directory / "dataset_test.txt";
  fs::path train_pose_path = directory / "dataset_train.txt";
  loadGtPosesFromFile(test_pose_path, is_street);
  loadGtPosesFromFile(train_pose_path, is_street);
}

void CambridgeLandmarksParser::loadGtPosesFromFile(fs::path pose_path,
                                                   bool is_street) {
  ifstream pose_file(pose_path.string());

  // first 3 lines are garbage
  string line;
  getline(pose_file, line);
  getline(pose_file, line);
  getline(pose_file, line);

  while (getline(pose_file, line)) {
    // TODO: will need to handle re-extracted content as a special case, since
    // some files will not have the same names as the original dataset.

    // seqY/frameXXXXX.png -> frame XXXXX-1 in dbsByName[seqY]
    // (or possible (XXXXX-1) * r for some arbitrary sampling rate

    // note: unlike tum, qw is first and not last
    // seq/filename x y z qw qx qy qz
    stringstream ss(line);
    string filename;
    float x, y, z, qw, qx, qy, qz;
    ss >> filename;
    read_float_or_nan_or_inf(ss, x);
    read_float_or_nan_or_inf(ss, y);
    read_float_or_nan_or_inf(ss, z);
    read_float_or_nan_or_inf(ss, qw);
    read_float_or_nan_or_inf(ss, qx);
    read_float_or_nan_or_inf(ss, qy);
    read_float_or_nan_or_inf(ss, qz);

    string seqdir(filename.substr(0, filename.find("/")));

    if (dbsByName.find(seqdir) == dbsByName.end()) {
      // we might only be processing a subset. that's fine.
      continue;
    }
    Database& cur_db = dbsByName.at(seqdir);
    int index = id_from_filename_and_dir(
        filename.substr(filename.find("/") + 1), seqdir, is_street);
    // IGNORE THE FOLLOWING LINE. THE ORIGINAL DATASET IS VERY UN-UNIFOMLY
    // SAMPLED.
    // FOR EXAMPLE, IN VIDEO 1, THE VIDEO FRAMES CORRESPOND TO THE LABELED
    // FRAMES
    // IN THE FOLLOWING WAY
    // 1 <--> 1
    // 2 <--> 2
    // 3 <--> 3
    // 9 <--> 4
    // 23 <--> 5
    // 38 <--> 6
    // SO 1/14 IS A VERY ROUGH APPROXIMATION, BUT IT'S NOT UNIFORM
    // for the most part, it seems like the original dataset took every 14th
    // frame. But some runs have fewer frames, so it's possible that they used
    // a non-integer sampling rate (and rounded to the nearest frame), or that
    // they dropped some frames.

    // OpenCV doesn't directly support quaternion conversions AFAIK
    Eigen::Quaternionf orientation(qw, qx, qy, qz);
    Eigen::Matrix3f as_rot = orientation.toRotationMatrix();
    Eigen::Vector3f translation(x, y, z);

    Mat R, t;
    eigen2cv(as_rot, R);
    eigen2cv(translation, t);

    // convert to 1/100th scale. DSO seems to be tuned for smaller scale
    // environments.
    t /= 100.0;

    // for some reason, the quaternion (but not the translation) seems to be
    // inverted? not sure what's up.
    R = R.t();

    rotationsAndTranslationsByDatabaseAndFrame[cur_db.db_id][index] =
        make_tuple(R, t);
  }
  pose_file.close();
}
bool CambridgeLandmarksParser::loadGroundTruthPose(const sdl::Frame& frame,
                                                   Mat& rotation,
                                                   Mat& translation) {
  auto& rotsAndTrans =
      rotationsAndTranslationsByDatabaseAndFrame.at(frame.dbId);
  if (rotsAndTrans.find(frame.index) == rotsAndTrans.end()) {
    return false;
  }
  rotation = get<0>(rotsAndTrans.at(frame.index));
  translation = get<1>(rotsAndTrans.at(frame.index));
  return true;
}

void SevenScenesParser::parseScene(
    vector<Database>& dbs,
    vector<pair<vector<reference_wrapper<Database>>, vector<Query>>>&
        dbs_with_queries,
    bool inject_ground_truth) {
  vector<fs::path> sequences;
  for (auto dir : fs::recursive_directory_iterator(directory)) {
    if (!fs::is_directory(dir)) {
      continue;
    }
    // each subdirectory should be of the form "seq-XX"
    if (dir.path().filename().string().find("seq") != 0) {
      continue;
    }
    sequences.push_back(dir);
  }
  if (sequences.empty()) {
    throw std::runtime_error("scene contained no sequences!");
  }

  sort(sequences.begin(), sequences.end());

  string line;
  vector<int> train_split, test_split;
  ifstream train_ifs((directory / "TrainSplit.txt").string());
  if (!train_ifs.good()) {
    throw runtime_error("Couldn't open TrainSplit.txt!");
  }
  while (train_ifs >> line) {
    int num(stoi(line.substr(8)));
    train_split.push_back(num);
  }
  ifstream test_ifs((directory / "TestSplit.txt").string());
  if (!test_ifs.good()) {
    throw runtime_error("Couldn't open TestSplit.txt!");
  }
  while (test_ifs >> line) {
    int num(stoi(line.substr(8)));
    test_split.push_back(num);
  }

  // create empty vecs to store the true training set and all the queries (we
  // can split train and test queries later)
  dbs_with_queries.emplace_back();

  // each sequence can produce 1 database and several queries
  for (const fs::path& sequence_dir : sequences) {
    dbs.emplace_back();

    Database& cur_db = dbs.back();
    cur_db.db_id = dbs.size() - 1;
    cur_db.setCachePath(cache / sequence_dir.filename());
  }

  for (unsigned int seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
    const fs::path& sequence_dir = sequences[seq_idx];
    Database& cur_db = dbs[seq_idx];

    bool in_train_set = false;
    for (int train_seq : train_split) {
      int num(stoi(sequence_dir.filename().string().substr(4)));
      if (train_seq == num) {
        in_train_set = true;
        break;
      }
    }
    if (in_train_set) {
      // store all the training databases in the first entry
      dbs_with_queries[0].first.push_back(cur_db);
    } else {
      // just create a pair with zero queries
      dbs_with_queries.emplace_back();
      dbs_with_queries.back().first.push_back(cur_db);
    }

    vector<string> sorted_images;
    for (auto file : fs::recursive_directory_iterator(sequence_dir)) {
      string name(file.path().filename().string());
      if (name.find(".color.png") == string::npos) {
        continue;
      }
      // these files are in the format frame-XXXXXX.color.png
      int id = stoi(name.substr(6, 6));
      //      if (id < 50) {
      //        continue;
      //      }
      unique_ptr<Frame> frame(new sdl::Frame(id, cur_db.db_id));
      frame->setImageLoader([file]() { return imread(file.path().string()); });
      frame->setPath(sequence_dir);
      string image_filename(file.path().string());
      frame->setCachePath(cache / sequence_dir.filename());
      sorted_images.push_back(image_filename);

      // store all the queries in the first entry
      // remove this conditional if you want to evaulate the train set on itself
      if (!in_train_set) {
        dbs_with_queries[0].second.emplace_back(cur_db.db_id, frame.get());
      }
      cur_db.addFrame(move(frame));
    }

    sort(sorted_images.begin(), sorted_images.end());
    // use calib from colmap
    fs::path calib_file(directory / "colmap-calib.txt");
    // TODO: rewrite the DSO interface to pass this calibration directly
    UndistortPinhole undistorter(calib_file.string().c_str(), false);
    dso::Mat33 K_eigen = undistorter.getK();
    Mat K(3, 3, CV_64F);
    eigen2cv(K_eigen, K);
    K.convertTo(K, CV_32F);
    int width = undistorter.getSize()[0];
    int height = undistorter.getSize()[1];
    // TODO: allow user to default to dummy configuration with optional
    // argument

    cur_db.setMapper(new DsoMapGenerator(K, width, height, sorted_images,
                                         cur_db.getCachePath().string(),
                                         !in_train_set));

    if (inject_ground_truth) {
      cur_db.setGtPoseLoader([this](const Frame& frame, Mat& R, Mat& t) {
        return loadGroundTruthPose(frame, R, t);
      });
    }
  }
}

bool SevenScenesParser::loadGroundTruthPose(const sdl::Frame& frame,
                                            Mat& rotation, Mat& translation) {
  // frame i -> frame-XXXXXi.pose.txt
  stringstream ss;
  ss << "frame-" << setfill('0') << setw(6) << frame.index << ".pose.txt";
  fs::path pose_path = frame.framePath / ss.str();

  ifstream ifs(pose_path.string());
  rotation.create(3, 3, CV_32FC1);
  translation.create(3, 1, CV_32FC1);
  // pose is stored as a 4x4 float matrix [R t; 0 1], so we only care about
  // the first 3 rows.
  for (int i = 0; i < 3; i++) {
    ifs >> rotation.at<float>(i, 0);
    ifs >> rotation.at<float>(i, 1);
    ifs >> rotation.at<float>(i, 2);
    ifs >> translation.at<float>(i, 0);
  }

  ifs.close();
  return true;
}
bool SevenScenesParser::loadGroundTruthDepth(const sdl::Frame& frame,
                                             cv::Mat& depth) {
  // frame i -> frame-XXXXXi.depth.png
  stringstream ss;
  ss << "frame-" << setfill('0') << setw(6) << frame.index << ".depth.png";
  depth = cv::imread((frame.framePath / ss.str()).string());

  return true;
}

void TumParser::parseScene(
    vector<Database>& dbs,
    vector<pair<vector<reference_wrapper<Database>>, vector<Query>>>&
        dbs_with_queries,
    bool inject_ground_truth) {
  vector<fs::path> sequences;
  for (auto dir : fs::recursive_directory_iterator(directory)) {
    if (!fs::is_directory(dir)) {
      continue;
    }
    // each subdirectory should be of the form "sequence_XX"
    if (dir.path().filename().string().find("sequence") != 0) {
      continue;
    }
    sequences.push_back(dir);
  }
  if (sequences.empty()) {
    throw std::runtime_error("scene contained no sequences!");
  }

  sort(sequences.begin(), sequences.end());

  // each sequence can produce 1 database and several queries
  for (const fs::path& sequence_dir : sequences) {
    dbs.emplace_back();

    Database& cur_db = dbs.back();
    cur_db.db_id = dbs.size() - 1;
    cur_db.setCachePath(cache / sequence_dir.filename());
  }

  for (unsigned int seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
    const fs::path& sequence_dir = sequences[seq_idx];
    Database& cur_db = dbs[seq_idx];
    cur_db.setTrainOnFirstHalf(true);

    dbs_with_queries.emplace_back();
    dbs_with_queries.back().first.push_back(cur_db);
    vector<Query>& queries = dbs_with_queries.back().second;

    fs::path image_zip = sequence_dir / "images.zip";
    // assumes all the images have been unzipped to a directory images/

    string calib = (sequence_dir / "camera.txt").string();
    string gamma_calib = (sequence_dir / "pcalib.txt").string();
    string vignette = (sequence_dir / "vignette.png").string();

    shared_ptr<ImageFolderReader> reader(new ImageFolderReader(
        image_zip.string(), calib, gamma_calib, vignette));
    for (int id = 0; id < reader->getNumImages(); ++id) {
      unique_ptr<Frame> frame(new sdl::Frame(id, cur_db.db_id));
      frame->setPath(sequence_dir);
      frame->setImageLoader([reader, id]() {
        ImageAndExposure* image = reader->getImage(id);
        Mat shallow(image->h, image->w, CV_32FC1, image->image);
        Mat converted;
        shallow.convertTo(converted, CV_8UC1);
        cvtColor(converted, converted, ColorConversionCodes::COLOR_GRAY2RGB);
        delete image;
        return converted;
      });
      frame->setCachePath(cache / sequence_dir.filename());

      queries.emplace_back(cur_db.db_id, frame.get());
      cur_db.addFrame(move(frame));
    }

    cur_db.setMapper(new DsoMapGenerator(sequence_dir.string()));
    if (inject_ground_truth) {
      cur_db.setGtPoseLoader([this](const Frame& frame, Mat& R, Mat& t) {
        return loadGroundTruthPose(frame, R, t);
      });
    }
    // preload the ground truth poses, so we don't have to seek every time
    // file is sequence_dir/groundtruthSync.txt
    fs::path pose_path = sequence_dir / "groundtruthSync.txt";
    ifstream pose_file(pose_path.string());

    string line;
    int index = 0;
    while (getline(pose_file, line)) {
      // images/XXXXX.jpg -> line XXXXX of groundtruthSync.txt (both are
      // 0-indexed)

      // time x y z qx qy qz qw
      stringstream ss(line);
      float time, x, y, z, qx, qy, qz, qw;
      ss >> time;
      read_float_or_nan_or_inf(ss, x);
      read_float_or_nan_or_inf(ss, y);
      read_float_or_nan_or_inf(ss, z);
      read_float_or_nan_or_inf(ss, qx);
      read_float_or_nan_or_inf(ss, qy);
      read_float_or_nan_or_inf(ss, qz);
      read_float_or_nan_or_inf(ss, qw);

      // OpenCV doesn't directly support quaternion conversions AFAIK
      Eigen::Quaternionf orientation(qw, qx, qy, qz);
      Eigen::Matrix3f as_rot = orientation.toRotationMatrix();
      Eigen::Vector3f translation(x, y, z);

      Mat R, t;
      eigen2cv(as_rot, R);
      eigen2cv(translation, t);

      rotationsAndTranslationsByDatabaseAndFrame[cur_db.db_id][index] =
          make_tuple(R, t);
      ++index;
    }
    pose_file.close();
  }
}

bool TumParser::loadGroundTruthPose(const sdl::Frame& frame, Mat& rotation,
                                    Mat& translation) {
  // for the TUM dataset, initial and final ground truth is available (I
  // assume during this time, the camera is tracked by an multi-view IR
  // tracker system or something), but otherwise the pose is recorded as NaN

  // It would almost certainly be better to load all the poses at once,
  // or at least to cache filehandles instead of re-opening the file for
  // each one. If/ opening/seeking/closing becomes a bottleneck, refactor
  // this.

  rotation = get<0>(rotationsAndTranslationsByDatabaseAndFrame.at(frame.dbId)
                        .at(frame.index));
  translation = get<1>(rotationsAndTranslationsByDatabaseAndFrame.at(frame.dbId)
                           .at(frame.index));
  return true;
}

}  // namespace sdl
