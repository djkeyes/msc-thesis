
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
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

void SevenScenesParser::parseScene(vector<sdl::Database>& dbs,
                                   vector<sdl::Query>& queries) {
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

  // each sequence can produce 1 database and several queries
  for (const fs::path& sequence_dir : sequences) {
    dbs.emplace_back();

    Database& cur_db = dbs.back();
    cur_db.db_id = dbs.size() - 1;
    if (!cache.empty()) {
      cur_db.setCachePath(cache / sequence_dir.filename());
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

      queries.emplace_back(cur_db.db_id, frame.get());
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
                                         cur_db.getCachePath().string()));
  }
}

void SevenScenesParser::loadGroundTruthPose(const sdl::Frame& frame,
                                            Mat& rotation, Mat& translation) {
  // frame i in db j -> seq-(j+1)/frame-XXXXXi.pose.txt
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
}

void TumParser::parseScene(vector<sdl::Database>& dbs,
                           vector<sdl::Query>& queries) {
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

void TumParser::loadGroundTruthPose(const sdl::Frame& frame, Mat& rotation,
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
}

}  // namespace sdl
