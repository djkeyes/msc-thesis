
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/QR"
#include "Eigen/StdVector"

#include "geometricburstiness/inverted_index.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

#include "FusedFeatureDescriptors.h"
#include "LargeBagOfWords.h"
#include "MapGenerator.h"
#include "Relocalization.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace fs = boost::filesystem;

// eigen magic
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(
    geometric_burstiness::QueryDescriptor<64>)

namespace sdl {

bool display_keypoints = false;

int max_image_index = 10000000;

fs::path Frame::getDescriptorFilename() const {
  stringstream ss;
  ss << "frame_" << setfill('0') << setw(6) << index
     << "_keypoints_and_descriptors.bin";
  fs::path filename = cachePath / ss.str();
  return filename;
}

bool Frame::loadDescriptors(vector<KeyPoint>& keypointsOut,
                            Mat& descriptorsOut) const {
  if (cachePath.empty()) {
    return false;
  }

  fs::path filename(getDescriptorFilename());

  if (!fs::exists(filename)) {
    return false;
  }

  ifstream ifs(filename.string(), ios_base::in | ios_base::binary);

  uint32_t num_descriptors, descriptor_size, data_type;
  ifs.read(reinterpret_cast<char*>(&num_descriptors), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&descriptor_size), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&data_type), sizeof(uint32_t));

  descriptorsOut.create(num_descriptors, descriptor_size, data_type);
  ifs.read(reinterpret_cast<char*>(descriptorsOut.data),
           num_descriptors * descriptor_size * descriptorsOut.elemSize());

  keypointsOut.reserve(num_descriptors);
  for (unsigned int i = 0; i < num_descriptors; i++) {
    float x, y, size, angle, response;
    uint32_t octave, class_id;
    ifs.read(reinterpret_cast<char*>(&x), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&y), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&size), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&angle), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&response), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&octave), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&class_id), sizeof(uint32_t));
    keypointsOut.emplace_back(x, y, size, angle, response, octave, class_id);
  }

  ifs.close();

  return true;
}
bool Frame::descriptorsExist() const {
  if (cachePath.empty()) {
    return false;
  }

  fs::path filename(getDescriptorFilename());

  if (!fs::exists(filename)) {
    return false;
  }

  ifstream ifs(filename.string(), ios_base::in | ios_base::binary);

  // could also check the size/datatype and compare to filesize
  return ifs.good();
}
int Frame::countDescriptors() const {
  if (cachePath.empty()) {
    return 0;
  }

  fs::path filename(getDescriptorFilename());

  if (!fs::exists(filename)) {
    return 0;
  }

  ifstream ifs(filename.string(), ios_base::in | ios_base::binary);

  if (!ifs.good()) {
    return 0;
  }

  uint32_t num_descriptors;
  ifs.read(reinterpret_cast<char*>(&num_descriptors), sizeof(uint32_t));
  ifs.close();
  return num_descriptors;
}
void Frame::saveDescriptors(const vector<KeyPoint>& keypoints,
                            const Mat& descriptors) const {
  if (cachePath.empty()) {
    return;
  }

  fs::path filename(getDescriptorFilename());
  // create directory if it doesn't exist
  fs::create_directories(filename.parent_path());

  ofstream ofs(filename.string(), ios_base::out | ios_base::binary);
  int descriptor_size = descriptors.size().width;
  int num_descriptors = descriptors.size().height;
  int data_type = descriptors.type();

  ofs.write(reinterpret_cast<const char*>(&num_descriptors), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&descriptor_size), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&data_type), sizeof(uint32_t));

  ofs.write(reinterpret_cast<const char*>(descriptors.data),
            num_descriptors * descriptor_size * descriptors.elemSize());
  for (int i = 0; i < num_descriptors; i++) {
    uint32_t octave(keypoints[i].octave);
    uint32_t class_id(keypoints[i].class_id);
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].pt.x), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].pt.y), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].size), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].angle),
              sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].response),
              sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&octave), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(&class_id), sizeof(uint32_t));
  }

  ofs.close();
}

boost::filesystem::path Frame::getSceneCoordinateFilename() const {
  stringstream ss;
  ss << "sparse_scene_coords_" << setfill('0') << setw(6) << index << ".bin";
  return cachePath / ss.str();
}
void Frame::saveSceneCoordinates(
    const SceneCoordinateMap& coordinate_map) const {
  fs::path filename(getSceneCoordinateFilename().string());
  fs::create_directories(filename.parent_path());

  ofstream ofs(filename.string(), ios_base::out | ios_base::binary);

  uint32_t rows = coordinate_map.height;
  uint32_t cols = coordinate_map.width;
  uint32_t size = coordinate_map.coords.size();
  ofs.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
  for (const auto& element : coordinate_map.coords) {
    const cv::Vec3f& val = element.second.coord;
    uint16_t row = element.first.first;
    uint16_t col = element.first.second;
    ofs.write(reinterpret_cast<const char*>(&row), sizeof(uint16_t));
    ofs.write(reinterpret_cast<const char*>(&col), sizeof(uint16_t));
    ofs.write(reinterpret_cast<const char*>(&val), sizeof(cv::Vec3f));

    // NOTE: THIS MAKES SAVED BINARIES INCOMPATIBLE WITH EACH OTHER. BE WARY.
    if (saveVariance) {
      assert(element.second.hasVariance);
      float inv_depth(element.second.inverseDepth);
      float inv_var(element.second.inverseVariance);
      SE3 cam2world(element.second.observerCam2World);

      ofs.write(reinterpret_cast<const char*>(&inv_depth), sizeof(float));
      ofs.write(reinterpret_cast<const char*>(&inv_var), sizeof(float));

      double x = cam2world.translation().x();
      double y = cam2world.translation().y();
      double z = cam2world.translation().z();
      double qx = cam2world.unit_quaternion().x();
      double qy = cam2world.unit_quaternion().y();
      double qz = cam2world.unit_quaternion().z();
      double qw = cam2world.unit_quaternion().w();

      ofs.write(reinterpret_cast<const char*>(&x), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&y), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&z), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&qx), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&qy), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&qz), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&qw), sizeof(double));
    }
  }

  ofs.close();
}
SceneCoordinateMap Frame::loadSceneCoordinates() const {
  if (!fs::exists(getSceneCoordinateFilename())) {
    return SceneCoordinateMap(2, 2);
  }
  ifstream ifs(getSceneCoordinateFilename().string(),
               ios_base::in | ios_base::binary);

  uint32_t rows, cols, size;
  ifs.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
  SceneCoordinateMap coordinate_map(rows, cols);
  for (int i = 0; i < static_cast<int>(size); ++i) {
    uint16_t row, col;
    ifs.read(reinterpret_cast<char*>(&row), sizeof(uint16_t));
    ifs.read(reinterpret_cast<char*>(&col), sizeof(uint16_t));
    cv::Vec3f val;
    ifs.read(reinterpret_cast<char*>(&val), sizeof(cv::Vec3f));

    SceneCoord coord;
    if (saveVariance) {
      float inv_depth,  inv_variance;

      ifs.read(reinterpret_cast<char*>(&inv_depth), sizeof(float));
      ifs.read(reinterpret_cast<char*>(&inv_variance), sizeof(float));

      double x, y, z, qx, qy, qz, qw;
      ifs.read(reinterpret_cast<char*>(&x), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&y), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&z), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&qx), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&qy), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&qz), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&qw), sizeof(double));

      Eigen::Quaterniond q(qw, qx, qy, qz);
      dso::SE3::Point t(x, y, z);
      SE3 cam2world(q, t);

      coord = SceneCoord(val, inv_depth, inv_variance, cam2world);
    } else {
      coord = SceneCoord(val);
    }
    coordinate_map.coords.insert(
        make_pair(make_pair(row, col), coord));
  }
  ifs.close();

  return coordinate_map;
}
fs::path Frame::getPoseFilename() const {
  stringstream ss;
  ss << "pose_from_vo_" << setfill('0') << setw(6) << index << ".bin";
  return cachePath / ss.str();
}
void Frame::savePose(const dso::SE3& pose) const {
  fs::path filename(getPoseFilename().string());
  fs::create_directories(filename.parent_path());

  ofstream ofs(filename.string(), ios_base::out | ios_base::binary);

  double x = pose.translation().x();
  double y = pose.translation().y();
  double z = pose.translation().z();
  double qx = pose.unit_quaternion().x();
  double qy = pose.unit_quaternion().y();
  double qz = pose.unit_quaternion().z();
  double qw = pose.unit_quaternion().w();

  ofs.write(reinterpret_cast<const char*>(&x), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&y), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&z), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&qx), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&qy), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&qz), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&qw), sizeof(double));
  ofs.close();
}
dso::SE3 Frame::loadPose() const {
  if (!fs::exists(getPoseFilename())) {
    double nan = numeric_limits<double>::quiet_NaN();
    return dso::SE3(Eigen::Quaterniond::Identity(),
                    dso::SE3::Point(nan, nan, nan));
  }
  ifstream ifs(getPoseFilename().string(), ios_base::in | ios_base::binary);

  double x, y, z, qx, qy, qz, qw;
  ifs.read(reinterpret_cast<char*>(&x), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&y), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&z), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&qx), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&qy), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&qz), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&qw), sizeof(double));
  ifs.close();

  Eigen::Quaterniond q(qw, qx, qy, qz);
  dso::SE3::Point t(x, y, z);
  return dso::SE3(q, t);
}
struct Database::InvertedIndexImpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  geometric_burstiness::InvertedIndex<64> invertedIndex;
};

Database::Database()
    : db_id(-1),
      trainOnFirstHalf(false),
      associateWithDepthMaps(false),
      frames(new map<int, unique_ptr<Frame>>()),
      pInvertedIndexImpl(new Database::InvertedIndexImpl()) {}
// Need to explicitly define destructor and move constructor, otherwise
// compiler can't handle unique_ptrs with forward-declared types.
Database::Database(Database&&) = default;
Database::~Database() = default;

vector<Result> Database::lookup(
    Query& query, unsigned int num_to_return,
    vector<reference_wrapper<Database>>& all_training_dbs) {
  vector<vector<int>> assignments;
  const Mat descriptors = query.computeDescriptors();
  bowExtractor->computeAssignments(descriptors, assignments, 1);

  const vector<KeyPoint>& keypoints = query.getKeypoints();
  int num_features = keypoints.size();

  int descriptor_size = descriptorExtractor->descriptorSize();

  vector<geometric_burstiness::QueryDescriptor<64>> query_descriptors(
      num_features);

  cout << "num features: " << num_features << endl;
  vector<int> ids;
  ids.reserve(num_features);
  for (int i = 0; i < num_features; ++i) {
    ids.push_back(i);
  }
  // if there's a ton of features, just subsample randomly
  if (num_features > 2000) {
    std::random_shuffle(ids.begin(), ids.end());
    ids.resize(2000);
    num_features = ids.size();
    cout << "resized to " << ids.size() << endl;
  }
  for (int j = 0; j < num_features; ++j) {
    int id = ids[j];
    if (descriptor_size == 128) {
      Map<MatrixXf> descriptor(
          reinterpret_cast<float*>(descriptors.row(id).data), descriptor_size,
          1);
      pInvertedIndexImpl->invertedIndex.PrepareQueryDescriptor(
          descriptor, &(query_descriptors[id]));
    } else if (descriptor_size >= 2 && descriptor_size < 128) {
      // not sure what to do if descriptor_size > 128
      Map<MatrixXf> descriptor(
          reinterpret_cast<float*>(descriptors.row(id).data), descriptor_size,
          1);

      // just 0-pad the descriptor
      Eigen::Matrix<float, 128, 1> padded_descriptor;
      int padding = 128 - descriptor_size;
      padded_descriptor << descriptor, Eigen::MatrixXf::Zero(padding, 1);
      pInvertedIndexImpl->invertedIndex.PrepareQueryDescriptor(
          padded_descriptor, &(query_descriptors[j]));
    } else {
      throw runtime_error(
          "Only descriptors in size range [2, 128] are supported.");
    }

    query_descriptors[j].relevant_word_ids.insert(
        query_descriptors[j].relevant_word_ids.end(), assignments[id].begin(),
        assignments[id].end());
    query_descriptors[j].x = keypoints[id].pt.x;
    query_descriptors[j].y = keypoints[id].pt.y;
    // this is irrellevant for us
    query_descriptors[j].a = 0;
    query_descriptors[j].b = 0;
    query_descriptors[j].c = 0;
    query_descriptors[j].feature_id = id;
  }
  for (int j = 0; j < num_features; ++j) {
    for (unsigned int k = 0; k < query_descriptors[j].relevant_word_ids.size();
         k++) {
      if (descriptor_size == 128) {
        query_descriptors[j].max_hamming_distance_per_word.push_back(32);
      } else if (descriptor_size < 128) {
        query_descriptors[j].max_hamming_distance_per_word.push_back(descriptor_size/4);
      }
    }
  }

  vector<geometric_burstiness::ImageScore> image_scores;
  pInvertedIndexImpl->invertedIndex.QueryIndex(query_descriptors,
                                               &image_scores);

  for (unsigned int i = 0; i < std::min<unsigned int>(50, image_scores.size());
       ++i) {
    geometric_burstiness::geometry::AffineFeatureMatches matches_fsm;
    pInvertedIndexImpl->invertedIndex.Get1To1MatchesForSpatialVerification(
        query_descriptors, image_scores[i], &matches_fsm);
  }

  vector<Result> results;
  for (unsigned int i = 0;
       i < min(static_cast<unsigned int>(image_scores.size()), num_to_return);
       ++i) {
    geometric_burstiness::geometry::AffineFeatureMatches matches_fsm;
    pInvertedIndexImpl->invertedIndex.Get1To1MatchesForSpatialVerification(
        query_descriptors, image_scores[i], &matches_fsm);

    int inverted_idx_id = image_scores[i].image_id;
    int db_id = inverted_idx_id / max_image_index;
    int frame_id = inverted_idx_id % max_image_index;

    bool found_db = false;
    for (Database& db : all_training_dbs) {
      if (static_cast<int>(db.db_id) == db_id) {
        found_db = true;
        results.emplace_back(*db.frames->at(frame_id));
        break;
      }
    }
    if (!found_db) {
      stringstream ss;
      ss << "Couldn't find corresponding db with id " << db_id
         << ". Current dbs have ids: [";
      for (Database& db : all_training_dbs) {
        ss << db.db_id << ", ";
      }
      ss << "].";
      throw runtime_error(ss.str());
    }

    for (const auto& correspondence : matches_fsm) {
      int query_feature_id = correspondence.feature1_.feature_id_;

      int db_feature_id = correspondence.features2_[0].feature_id_;

      results.back().matches.emplace_back(query_feature_id, db_feature_id,
                                          image_scores[i].image_id, 1.0);
    }
  }

  return results;
}

void Database::setupFeatureDetector(bool associate_with_depth_maps) {
  associateWithDepthMaps = associate_with_depth_maps;
}

void Database::setDescriptorExtractor(
    Ptr<DescriptorExtractor> descriptor_extractor) {
  descriptorExtractor = descriptor_extractor;
}
void Database::setBowExtractor(
    Ptr<BOWSparseImgDescriptorExtractor> bow_extractor) {
  bowExtractor = bow_extractor;
}

void Database::onlyDoMapping() {
  if (gtPoseLoader) {
    // if provided, use the ground truth pose to seed DSO
    mapGen->runVisualOdometry([this](int id) {
      if (frames->find(id) == frames->end()) {
        cout << "couldn't find frame id " << id << endl;
        assert(false);
        return unique_ptr<SE3>(nullptr);
      } else {
        cv::Mat R, t;
        if (gtPoseLoader(*frames->at(id), R, t)) {
          R.convertTo(R, CV_64F);
          t.convertTo(t, CV_64F);
          Eigen::Matrix3d R_eigen;
          Eigen::Vector3d t_eigen;
          cv2eigen(R, R_eigen);
          cv2eigen(t, t_eigen);
          SE3 pose(R_eigen, t_eigen);

          return unique_ptr<SE3>(new SE3(pose));
        } else {
          return unique_ptr<SE3>(nullptr);
        }
      }
    });
  } else {
    // just run vanilla DSO, hopefully no drift
    mapGen->runVisualOdometry();
  }
}
void Database::doMapping() {
  if (!needToRecomputeSceneCoordinates()) {
    return;
  }
  // first run the full mapping algorithm
  onlyDoMapping();

  // then fetch the depth maps / poses, and convert them to 2D -> 3D image maps
  map<int, SceneCoordinateMap> scene_coordinate_maps =
      mapGen->getSceneCoordinateMaps();
  std::map<int, dso::SE3>* poses = mapGen->getPoses();

  int width = scene_coordinate_maps.begin()->second.width;
  int height = scene_coordinate_maps.begin()->second.height;
  for (auto& element : *frames) {
    Frame& cur_frame = *element.second;
    int frame_id = element.first;

    auto iter = scene_coordinate_maps.find(frame_id);
    if (iter == scene_coordinate_maps.end()) {
      cur_frame.saveSceneCoordinates(SceneCoordinateMap(height, width));
    } else {
      cur_frame.saveSceneCoordinates(iter->second);
    }

    auto pose_iter = poses->find(frame_id);
    if (pose_iter == poses->end()) {
      double nan = numeric_limits<double>::quiet_NaN();
      dso::SE3 invalid(dso::SE3(Eigen::Quaterniond::Identity(),
                                dso::SE3::Point(nan, nan, nan)));
      cur_frame.savePose(invalid);
    } else {
      cur_frame.savePose(pose_iter->second);
    }
  }

  mapGen->saveCameraAdjacencyList((getCachePath() / "adjlist.txt").string());

  // Also save a pointcloud, for debugging
  mapGen->savePointCloudAsPcd((cachePath / "pointcloud.pcd").string());
  mapGen->savePointCloudAsPly((cachePath / "pointcloud.ply").string());

  // clear mapGen, since it hogs a bunch of memory
  mapGen.reset();
}
bool Database::needToRecomputeSceneCoordinates() const {
  unsigned int num_saved_coords = 0;
  for (auto file : fs::recursive_directory_iterator(cachePath)) {
    if (file.path().filename().string().find("sparse_scene_coords") !=
        string::npos) {
      num_saved_coords++;
    }
  }
  return num_saved_coords == 0;
}
bool Database::hasCachedDescriptors() const {
  for (const auto& element : *frames) {
    if (!element.second->descriptorsExist()) {
      return false;
    }
  }
  return true;
}
int Database::computeDescriptorsForEachFrame(
    map<int, vector<KeyPoint>>& image_keypoints,
    map<int, Mat>& image_descriptors) {
  int total_descriptors = 0;

  int num_displayed = 0;
  int total_frames = frames->size();
  int num_processed = 0;
  // iterate through the images
  for (const auto& element : *frames) {
    const auto& frame = element.second;

    image_keypoints[frame->index] = vector<KeyPoint>();
    image_descriptors.insert(make_pair(frame->index, Mat()));

    // this can be quite slow, so reload a cached copy from disk if it's
    // available
    if (!frame->loadDescriptors(image_keypoints[frame->index],
                                image_descriptors[frame->index])) {
      Mat colorImage = frame->imageLoader();
      if (associateWithDepthMaps) {
        SceneCoordFeatureDetector* with_depth =
            dynamic_cast<SceneCoordFeatureDetector*>(descriptorExtractor.get());
        // careful: if we're only using keypoints detected by visual
        // odometry, some frames may have 0 keypoints (ie due to dropped
        // frames or mapping failure (even if it recovered in later frames))
        SceneCoordinateMap scene_coords = frame->loadSceneCoordinates();

        with_depth->setCurrentSceneCoords(scene_coords);
      }
      descriptorExtractor->detect(colorImage, image_keypoints[frame->index]);
      descriptorExtractor->compute(colorImage, image_keypoints[frame->index],
                                   image_descriptors[frame->index]);

      if (display_keypoints && num_displayed < 5 &&
          !image_keypoints[frame->index].empty()) {
        // can use DrawMatchesFlags::DRAW_RICH_KEYPOINTS if we're displaying
        // SIFT
        drawKeypoints(colorImage, image_keypoints[frame->index], colorImage,
                      Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        stringstream ss;
        ss << "Keypoints in frame " << frame->index << " (db=" << frame->dbId
           << ")";
        string window_name(ss.str());
        namedWindow(window_name, WINDOW_AUTOSIZE);
        imshow(window_name, colorImage);
        waitKey(0);
        destroyWindow(window_name);

        ++num_displayed;
      }

      frame->saveDescriptors(image_keypoints[frame->index],
                             image_descriptors[frame->index]);
    }
    total_descriptors += image_descriptors[frame->index].rows;

    if (num_processed % 100 == 0) {
      cout << "\r " << num_processed << "/" << total_frames << " ("
           << setprecision(4)
           << static_cast<double>(num_processed) / total_frames * 100. << "%)"
           << flush;
    }
    ++num_processed;
  }
  cout << endl;
  return total_descriptors;
}

int Database::loadAllDescriptors(
    map<int, map<int, vector<KeyPoint>>>& image_keypoints,
    map<int, map<int, Mat>>& image_descriptors,
    vector<reference_wrapper<Database>>& all_training_dbs) {
  int total_descriptors = 0;
  // iterate through the images
  for (Database& db : all_training_dbs) {
    total_descriptors += db.computeDescriptorsForEachFrame(
        image_keypoints[db.db_id], image_descriptors[db.db_id]);
  }
  return total_descriptors;
}
fs::path Database::mergedDescriptorsFileFlag() {
  return getCachePath() / "merged";
}
bool Database::hasCoobservedDescriptors() {
  return fs::exists(mergedDescriptorsFileFlag());
}
void Database::mergeCoobservedDescriptors(
    const map<int, vector<KeyPoint>>& image_keypoints,
    const map<int, Mat>& image_descriptors) {
  // We could do lots of bookkeeping during map generation.
  // Alternatively, here's an easier way: for every pair of images with any
  // co-observed points, iterate through each scene coordinate and see if they
  // project to the same thing in both images.
  // This has the downside that each frame computes the average independently.

  ifstream adj_file((getCachePath() / "adjlist.txt").string());
  if (!adj_file.good()) {
    throw runtime_error("Adjacency list not found!");
  }

  map<int, set<int>> adjlist;
  int count;
  adj_file >> count;
  vector<int> kf_ids(count);
  for (int i = 0; i < count; ++i) {
    adj_file >> kf_ids[i];
  }
  for (int i = 0; i < count; ++i) {
    int num_adj;
    adj_file >> num_adj;
    for (int j = 0; j < num_adj; ++j) {
      int val;
      adj_file >> val;
      adjlist[kf_ids[i]].insert(val);
    }
  }
  // also add first neighbors (these adjacency lists don't correctly store
  // cliques of adjacent frames. I think. Should test this.)
  map<int, set<int>> nextadjlist;
  for (int k : kf_ids) {
    for (int adj : adjlist[k]) {
      nextadjlist[k].insert(adjlist[adj].begin(), adjlist[adj].end());
    }
    nextadjlist[k].erase(k);
  }
  for (int k : kf_ids) {
    adjlist[k].insert(nextadjlist[k].begin(), nextadjlist[k].end());
  }

  map<int, map<pair<int, int>, tuple<cv::Vec3f, Mat, int>>>
      frames_to_coords_to_3dpt_and_descr_sums;
  map<int, map<pair<int, int>, Mat>> frames_to_coords_to_orig_descr;
  for (auto& element : adjlist) {
    int a = element.first;

    if (frames->find(a) == frames->end() ||
        image_keypoints.find(a) == image_keypoints.end() ||
        image_descriptors.find(a) == image_descriptors.end()) {
      cout << "couldn't find frame a=" << a << "!" << endl;
      continue;
    }

    // need: keypoint, scene coords, descriptors
    SceneCoordinateMap scene_coords_a =
        frames->find(a)->second->loadSceneCoordinates();

    // first project a onto itself
    const vector<KeyPoint>& kpts_a = image_keypoints.find(a)->second;
    for (unsigned int k = 0; k < kpts_a.size(); ++k) {
      const KeyPoint& kpt = kpts_a[k];
      Mat descriptor = image_descriptors.at(a).row(k);
      pair<int, int> coord =
          make_pair(static_cast<int>(kpt.pt.y), static_cast<int>(kpt.pt.x));
      cv::Vec3f scene_coord(
          scene_coords_a.coords.at(make_pair(coord.first, coord.second)).coord);
      frames_to_coords_to_3dpt_and_descr_sums[a][coord] =
          make_tuple(scene_coord, descriptor.clone(), 1);
      frames_to_coords_to_orig_descr[a][coord] = descriptor;
    }
  }

  Mat K = getCalibration();
  float fx = K.at<float>(0, 0);
  float fy = K.at<float>(1, 1);
  float cx = K.at<float>(0, 2);
  float cy = K.at<float>(1, 2);

  for (auto& element : adjlist) {
    int a = element.first;

    for (int b : element.second) {
      if (frames->find(b) == frames->end() ||
          frames_to_coords_to_orig_descr.find(b) ==
              frames_to_coords_to_orig_descr.end()) {
        cout << "couldn't find frame b=" << b << "!" << endl;
        continue;
      }

      Frame* frame_b = frames->find(b)->second.get();
      SceneCoordinateMap scene_coords_b = frame_b->loadSceneCoordinates();
      SE3 pose(frame_b->loadPose());

      int num_in_common = 0;
      // project each 3d point in a into b
      for (auto& coords_3dpt_and_descr_sum_a :
           frames_to_coords_to_3dpt_and_descr_sums[a]) {
        tuple<cv::Vec3f, Mat, int>& scenecoord_descr_sum =
            coords_3dpt_and_descr_sum_a.second;

        cv::Vec3f scene_coord_a = get<0>(scenecoord_descr_sum);
        Eigen::Vector3f scene_coord_a_eigen;
        cv2eigen(scene_coord_a, scene_coord_a_eigen);

        auto local = pose.inverse() * scene_coord_a_eigen.cast<double>();

        if (local.z() <= 0) {
          continue;
        }
        int u = static_cast<int>(round(local.x() / local.z() * fx + cx));
        int v = static_cast<int>(round(local.y() / local.z() * fy + cy));

        if (u < 0 || v < 0 || u >= scene_coords_b.width ||
            v >= scene_coords_b.height) {
          continue;
        }

        auto b_coord_iter = scene_coords_b.coords.find(make_pair(v, u));
        if (b_coord_iter == scene_coords_b.coords.end()) {
          continue;
        }

        const cv::Vec3f scene_coord_b = b_coord_iter->second.coord;
        if (cv::norm(scene_coord_a - scene_coord_b) > 0.0001) {
          continue;
        }

        auto iter = frames_to_coords_to_orig_descr[b].find(make_pair(v, u));
        if (iter == frames_to_coords_to_orig_descr[b].end()) {
          // for whatever reason, we may have chosen not to evaluate the
          // descriptor at this point.
          continue;
        }

        get<1>(scenecoord_descr_sum) += iter->second;
        get<2>(scenecoord_descr_sum)++;

        num_in_common++;
      }
    }
  }

  // divide and re-assign
  for (auto& element : frames_to_coords_to_3dpt_and_descr_sums) {
    int index = element.first;
    Frame* frame = frames->at(index).get();

    const vector<KeyPoint>& kpts = image_keypoints.find(index)->second;
    Mat merged(image_descriptors.at(index).size(),
               image_descriptors.at(index).type());
    for (unsigned int k = 0; k < kpts.size(); ++k) {
      const KeyPoint& kpt = kpts[k];

      pair<int, int> coord =
          make_pair(static_cast<int>(kpt.pt.y), static_cast<int>(kpt.pt.x));

      tuple<cv::Vec3f, Mat, int>& scenecoord_descr_sum =
          element.second.at(coord);
      merged.row(k) = get<1>(scenecoord_descr_sum) /
                      static_cast<float>(get<2>(scenecoord_descr_sum));
    }
    frame->saveDescriptors(kpts, merged);
  }

  ofstream ofs(mergedDescriptorsFileFlag().string());
  ofs.close();
}

void Database::doClustering(
    const map<int, map<int, Mat>>& image_descriptors,
    vector<reference_wrapper<Database>>& all_train_dbs) {
  if (!loadVocabulary(vocabulary)) {
    int max_iters = 5;
    TermCriteria terminate_criterion(TermCriteria::MAX_ITER, max_iters, 0.0);
    BOWApproxKMeansTrainer bow_trainer(vocabulary_size, terminate_criterion);

    for (Database& db : all_train_dbs) {
      for (const auto& element : *db.frames) {
        int index = element.second->index;
        if (trainOnFirstHalf &&
            static_cast<unsigned int>(index) > db.frames->size() / 2) {
          continue;
        }
        if (!image_descriptors.at(db.db_id).at(index).empty()) {
          int descriptor_count = image_descriptors.at(db.db_id).at(index).rows;

          for (int i = 0; i < descriptor_count; i++) {
            bow_trainer.add(image_descriptors.at(db.db_id).at(index).row(i));
          }
        }
      }
    }
    vocabulary = bow_trainer.cluster();
    for (Database& db : all_train_dbs) {
      db.saveVocabulary(vocabulary);
    }
  }
  for (Database& db : all_train_dbs) {
    db.bowExtractor->setVocabulary(vocabulary);
  }
}

map<int, vector<int>> Database::computeBowDescriptors(
    const map<int, Mat>& image_descriptors) {
  // Create training data by converting each keyframe to a bag of words
  map<int, vector<int>> assignments;
  for (const auto& element : *frames) {
    int index = element.second->index;
    if (trainOnFirstHalf &&
        static_cast<unsigned int>(index) > frames->size() / 2) {
      continue;
    }

    vector<vector<int>> assignment_singletons;
    bowExtractor->computeAssignments(image_descriptors.at(index),
                                     assignment_singletons);
    assignments[index].reserve(assignment_singletons.size());
    for (unsigned int i = 0; i < assignment_singletons.size(); ++i) {
      assignments[index].push_back(assignment_singletons[i][0]);
    }
  }
  return assignments;
}

MatrixXf Database::generateRandomProjection(int num_rows) {
  default_random_engine generator;
  normal_distribution<float> distribution(0.0, 1.0);

  int descriptor_size = 128;
  MatrixXf random_matrix(descriptor_size, descriptor_size);
  for (int i = 0; i < descriptor_size; i++) {
    for (int j = 0; j < descriptor_size; j++) {
      random_matrix(i, j) = distribution(generator);
    }
  }
  ColPivHouseholderQR<MatrixXf> qr_decomp(random_matrix);
  MatrixXf Q = qr_decomp.householderQ();
  MatrixXf projection = Q.topRows(num_rows);
  return projection;
}
MatrixXf Database::computeHammingThresholds(
    const MatrixXf& projection_matrix,
    const map<int, map<int, Mat>>& image_descriptors,
    const map<int, map<int, vector<int>>> descriptor_assignments,
    vector<reference_wrapper<Database>>& all_training_dbs) {
  Matrix<float, 64, Eigen::Dynamic> hamming_thresholds;
  hamming_thresholds.resize(64, vocabulary_size);
  int descriptor_size = descriptorExtractor->descriptorSize();

  int num_images = 0;
  for (auto& descriptor_map : image_descriptors) {
    num_images += static_cast<int>(descriptor_map.second.size());
  }
  cout << " Found " << num_images << " database images " << endl;

  // Loads for each word up to 10k nearby descriptors and then computes the
  // thresholds.
  vector<vector<vector<float>>> entries_per_word(vocabulary_size);
  for (int i = 0; i < vocabulary_size; ++i) {
    entries_per_word[i].resize(64);
    for (int j = 0; j < 64; ++j) entries_per_word[i][j].clear();
  }
  vector<int> num_desc_per_word(vocabulary_size, 0);
  int num_missing_words = vocabulary_size;

  vector<pair<reference_wrapper<Database>, int>> randomly_permuted_db_ids;
  randomly_permuted_db_ids.reserve(num_images);
  for (Database& db : all_training_dbs) {
    for (const auto& element : image_descriptors.at(db.db_id)) {
      if (trainOnFirstHalf &&
          static_cast<unsigned int>(element.first) > db.frames->size() / 2) {
        continue;
      }
      pair<reference_wrapper<Database>, int> pair(db, element.first);
      randomly_permuted_db_ids.push_back(pair);
    }
  }
  random_shuffle(randomly_permuted_db_ids.begin(),
                 randomly_permuted_db_ids.end());

  cout << " Determining relevant images per word " << endl;
  const int kNumDesiredDesc = 10000;
  for (unsigned int i = 0; i < randomly_permuted_db_ids.size(); ++i) {
    if (num_missing_words == 0) break;

    Database& db = randomly_permuted_db_ids[i].first;
    int id = randomly_permuted_db_ids[i].second;

    const Mat descriptors = image_descriptors.at(db.db_id).at(id);

    int num_features = descriptors.rows;

    if (num_features == 0) {
      continue;
    }

    for (int j = 0; j < num_features; ++j) {
      const int closest_word = descriptor_assignments.at(db.db_id).at(id)[j];
      if (num_desc_per_word[closest_word] >= kNumDesiredDesc) {
        continue;
      }

      // map the opencv memory to an eigen matrix
      // Ostensibly the opencv matrix is 1xD, and the eigen matrix needs
      // to be Dx1, but it doesn't really matter since it's 1-dimensional
      // and stored contiguously.

      Eigen::Matrix<float, 64, 1> proj;
      if (descriptor_size == 128) {
        Map<MatrixXf> descriptor(
            reinterpret_cast<float*>(descriptors.row(j).data), descriptor_size,
            1);
        proj = projection_matrix * descriptor;

      } else if (descriptor_size >= 2 && descriptor_size < 128) {
        // not sure what to do if descriptor_size > 128
        Map<MatrixXf> descriptor(
            reinterpret_cast<float*>(descriptors.row(j).data), descriptor_size,
            1);

        // just 0-pad the descriptor
        Eigen::Matrix<float, 128, 1> padded_descriptor;
        int padding = 128 - descriptor_size;
        padded_descriptor << descriptor, Eigen::MatrixXf::Zero(padding, 1);
        proj = projection_matrix * padded_descriptor;
      } else {
        throw runtime_error(
            "Only descriptors in size range [2, 128] are supported.");
      }

      for (int k = 0; k < 64; ++k) {
        entries_per_word[closest_word][k].push_back(proj[k]);
      }
      num_desc_per_word[closest_word] += 1;

      if (num_desc_per_word[closest_word] == kNumDesiredDesc) {
        --num_missing_words;
      }
    }

    if (i % 100 == 0) {
      cout << "\r " << i << flush;
    }
  }
  cout << endl;

  // For each word, computes the thresholds.
  cout << " Computing the thresholds per word " << endl;
  for (int i = 0; i < vocabulary_size; ++i) {
    int num_desc = num_desc_per_word[i];

    if (num_desc == 0) {
      cout << " WARNING: FOUND EMPTY WORD " << i << endl;
      hamming_thresholds.col(i) = Eigen::Matrix<float, 64, 1>::Zero();
    } else {
      const int median_element = num_desc / 2;
      for (int k = 0; k < 64; ++k) {
        nth_element(entries_per_word[i][k].begin(),
                    entries_per_word[i][k].begin() + median_element,
                    entries_per_word[i][k].end());
        hamming_thresholds(k, i) = entries_per_word[i][k][median_element];
      }
    }

    if (i % 1000 == 0) {
      cout << "\r word " << i << flush;
    }
  }
  cout << " done" << endl;

  return hamming_thresholds;
}
fs::path Database::getInvertedIndexFilename() const {
  return cachePath / "invertedIndex.bin";
}
fs::path Database::getInvertedIndexWeightsFilename() const {
  return cachePath / "invertedIndex.bin.weights";
}
bool Database::loadInvertedIndex(InvertedIndexImpl& inverted_index_impl) const {
  if (cachePath.empty()) {
    return false;
  }

  fs::path filename(getInvertedIndexFilename());
  fs::path weights_filename(getInvertedIndexWeightsFilename());

  if (!fs::exists(filename) && !fs::exists(weights_filename)) {
    return false;
  }

  bool success =
      inverted_index_impl.invertedIndex.LoadInvertedIndex(filename.string());
  success &= inverted_index_impl.invertedIndex.ReadWeightsAndConstants(
      weights_filename.string());
  return success;
}
void Database::saveInvertedIndex(
    const InvertedIndexImpl& inverted_index_impl) const {
  if (cachePath.empty()) {
    return;
  }

  fs::path filename(getInvertedIndexFilename());
  fs::create_directories(filename.parent_path());

  fs::path weights_filename(getInvertedIndexWeightsFilename());

  inverted_index_impl.invertedIndex.SaveInvertedIndex(filename.string());
  inverted_index_impl.invertedIndex.SaveWeightsAndConstants(
      weights_filename.string());
}

void Database::buildInvertedIndex(
    const map<int, map<int, vector<KeyPoint>>>& image_keypoints,
    const map<int, map<int, Mat>>& image_descriptors,
    vector<reference_wrapper<Database>>& all_training_dbs) {
  cout << "Computing bow descriptors for each image in training set using "
          "nearest neighbor to each descriptor..."
       << endl;
  map<int, map<int, vector<int>>> descriptor_assignments;
  for (Database& db : all_training_dbs) {
    descriptor_assignments[db.db_id] =
        db.computeBowDescriptors(image_descriptors.at(db.db_id));
  }
  cout << "Finished computing bags of words." << endl;

  MatrixXf projection_matrix = generateRandomProjection(64);
  MatrixXf hamming_thresholds =
      computeHammingThresholds(projection_matrix, image_descriptors,
                               descriptor_assignments, all_training_dbs);

  geometric_burstiness::InvertedIndex<64>& inverted_index =
      pInvertedIndexImpl->invertedIndex;
  cout << "initializing index to vocabulary size " << vocabulary_size << endl;
  inverted_index.InitializeIndex(vocabulary_size);
  cout << "finished initialization" << endl;

  inverted_index.SetProjectionMatrix(projection_matrix);
  inverted_index.SetHammingThresholds(hamming_thresholds);

  int descriptor_size = descriptorExtractor->descriptorSize();

  int total_number_entries = 0;
  for (Database& db : all_training_dbs) {
    for (const auto& element : *db.frames) {
      int index = element.second->index;
      if (trainOnFirstHalf &&
          static_cast<unsigned int>(index) > db.frames->size() / 2) {
        continue;
      }
      const vector<cv::KeyPoint>& keypoints =
          image_keypoints.at(db.db_id).at(index);
      const Mat& descriptors = image_descriptors.at(db.db_id).at(index);

      int num_features = static_cast<int>(keypoints.size());

      if (num_features == 0) {
        continue;
      }

      for (int j = 0; j < num_features; ++j, ++total_number_entries) {
        geometric_burstiness::InvertedFileEntry<64> entry;
        if (index >= max_image_index) {
          stringstream ss;
          ss << "Image index is too large to hash together with database id. "
             << index << " >= " << max_image_index;
          throw runtime_error(ss.str());
        }
        entry.image_id = db.db_id * max_image_index + index;
        entry.feature_id = j;
        entry.x = keypoints[j].pt.x;
        entry.y = keypoints[j].pt.y;
        // TODO(daniel): These are geometric properties of affine
        // descriptors. I think these are only used during geometric-based
        // reranking, so maybe it's okay to leave them empty. For regular
        // SIFT, it might be possible to compute these values, but I don't
        // see how this would be computable for a general descriptor.
        entry.a = 0;
        entry.b = 0;
        entry.c = 0;

        const int closest_word =
            descriptor_assignments.at(db.db_id).at(index)[j];
        if (closest_word < 0 || closest_word >= vocabulary_size) {
          throw runtime_error("Impossible word " + closest_word);
        }

        Map<MatrixXf> descriptor(
            reinterpret_cast<float*>(descriptors.row(j).data), descriptor_size,
            1);

        // geometric_burstiness::InvertedIndex expects descriptors to be size
        // 128
        // :(
        if (descriptor_size == 128) {
          inverted_index.AddEntry(entry, closest_word, descriptor);
        } else if (descriptor_size >= 2 && descriptor_size < 128) {
          // not sure what to do if descriptor_size > 128

          // just 0-pad the descriptor
          Eigen::Matrix<float, 128, 1> padded_descriptor;
          int padding = 128 - descriptor_size;
          padded_descriptor << descriptor, Eigen::MatrixXf::Zero(padding, 1);
          inverted_index.AddEntry(entry, closest_word, padded_descriptor);
        } else {
          throw runtime_error(
              "Only descriptors in size range [2, 128] are supported.");
        }
      }
    }
  }
  cout << " The index contains " << total_number_entries << " entries "
       << "in total" << endl;

  cout << " Estimates the descriptor space densities " << endl;
  inverted_index.EstimateDescriptorSpaceDensities();

  inverted_index.FinalizeIndex();
  cout << " Inverted index finalized" << endl;

  cout << "Computing weights and constants" << endl;
  inverted_index.ComputeWeightsAndNormalizationConstants();

  for (Database& db : all_training_dbs) {
    db.saveInvertedIndex(*pInvertedIndexImpl);
  }
}

void Database::runVoAndComputeDescriptors() {
  cout << "Training database " << db_id << ", saving to "
       << getCachePath().string() << endl;

  // create cache dir if it doesn't already exist
  fs::create_directories(cachePath);

  if (mapGen) {
    cout << "Mapping environment using specified SLAM system..." << endl;
    doMapping();
    cout << "Mapping complete." << endl;
  } else {
    cout << "Skipping mapping step, because not SLAM system was specified."
         << endl;
  }

  map<int, vector<KeyPoint>> image_keypoints;
  map<int, Mat> image_descriptors;

  // check if we haven't compute the descriptors yet
  if (!hasCachedDescriptors() || !hasCoobservedDescriptors()) {
    cout << "computing descriptors for each keyframe..." << endl;
    int descriptor_count =
        computeDescriptorsForEachFrame(image_keypoints, image_descriptors);
    cout << "computed " << descriptor_count << " descriptors in "
         << frames->size() << " frames." << endl;
    // TODO: might want to disable this for databases containing test queries.
    // or try it both ways and see the outcome.
    mergeCoobservedDescriptors(image_keypoints, image_descriptors);
  } else {
    cout << "Already have cached descriptors." << endl;
  }
}
void Database::assignAndBuildIndex(
    vector<reference_wrapper<Database>>& all_training_dbs) {
  // Descriptors should have been computed as a precondition. Load them if we
  // haven't clustered the vocabulary yet, or if we need to compute word
  // assignments for the inverted index.

  map<int, map<int, vector<KeyPoint>>> image_keypoints;
  map<int, map<int, Mat>> image_descriptors;

  bool has_index = loadInvertedIndex(*pInvertedIndexImpl);
  if (!has_index || !hasCachedVocabularyFile()) {
    cout << "loading descriptors for each keyframe..." << endl;
    int descriptor_count = loadAllDescriptors(
        image_keypoints, image_descriptors, all_training_dbs);
    int total_frames = 0;
    for (Database& db : all_training_dbs) {
      total_frames += db.frames->size();
    }
    cout << "loaded " << descriptor_count << " descriptors in " << total_frames
         << " frames." << endl;
  } else {
    cout << "Already have inverted index." << endl;
  }

  // even if the vocab is already cached, we need to load it so we can find NNs
  // for queries
  cout << "Training vocabulary..." << endl;
  doClustering(image_descriptors, all_training_dbs);
  cout << "Finished training vocabulary." << endl;

  if (!has_index) {
    cout << "Building inverted index..." << endl;
    buildInvertedIndex(image_keypoints, image_descriptors, all_training_dbs);
    cout << "Finished inverted index." << endl;
  }
}

void Database::addFrame(unique_ptr<Frame> frame) {
  int index = frame->index;
  frames->insert(make_pair(index, move(frame)));
}

fs::path Database::getVocabularyFilename() const {
  return cachePath / "clusters.bin";
}
bool Database::loadVocabulary(cv::Mat& vocabularyOut) const {
  if (cachePath.empty()) {
    return false;
  }

  fs::path filename(getVocabularyFilename());

  if (!fs::exists(filename)) {
    return false;
  }

  ifstream ifs(filename.string(), ios_base::in | ios_base::binary);

  uint32_t rows, cols, size, type;
  ifs.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&type), sizeof(uint32_t));
  vocabularyOut.create(rows, cols, type);
  ifs.read(reinterpret_cast<char*>(vocabularyOut.data), rows * cols * size);

  ifs.close();

  return true;
}
void Database::saveVocabulary(const cv::Mat& vocabulary) const {
  if (cachePath.empty()) {
    return;
  }

  fs::path filename(getVocabularyFilename());
  // create directory if it doesn't exist
  fs::create_directories(filename.parent_path());

  ofstream ofs(filename.string(), ios_base::out | ios_base::binary);

  uint32_t size(vocabulary.elemSize());
  uint32_t type(vocabulary.type());
  ofs.write(reinterpret_cast<const char*>(&vocabulary.rows), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&vocabulary.cols), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&type), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(vocabulary.data),
            vocabulary.rows * vocabulary.cols * size);

  ofs.close();
}
void Database::copyVocabularyFileFrom(const Database& from) const {
  // TODO: a cache path is basically required for most of this pipeline to
  // work. Make that a required argument.

  fs::path filename(getVocabularyFilename());
  // create directory if it doesn't exist
  fs::create_directories(filename.parent_path());

  fs::copy_file(from.getVocabularyFilename(), filename);
}

bool Database::hasCachedVocabularyFile() const {
  return fs::exists(getVocabularyFilename());
}
Query::Query(const unsigned int parent_database_id, const Frame* frame)
    : parent_database_id(parent_database_id),
      frame(frame),
      detectFromDepthMaps(false) {}

void Query::setDescriptorExtractor(
    Ptr<DescriptorExtractor> descriptor_extractor) {
  descriptorExtractor = descriptor_extractor;
}

void Query::setupFeatureDetector(bool detect_from_depth_maps) {
  detectFromDepthMaps = detect_from_depth_maps;
}

const Mat Query::readColorImage() const { return frame->imageLoader(); }
const Mat Query::computeDescriptors() {
  Mat descriptors;
  keypoints.clear();
  if (!frame->loadDescriptors(keypoints, descriptors)) {
    Mat colorImage = frame->imageLoader();

    if (detectFromDepthMaps) {
      throw runtime_error(
          "computeFeatures() for query with 3D points "
          "not yet implemented! Need to look up the associated "
          "database and call loadSceneCoordinates().");
    } else {
      descriptorExtractor->detect(colorImage, keypoints);
    }
    descriptorExtractor->compute(colorImage, keypoints, descriptors);
    frame->saveDescriptors(keypoints, descriptors);
  }
  return descriptors;
}

const vector<KeyPoint>& Query::getKeypoints() const { return keypoints; }

map<int, list<DMatch>> doRatioTest(Mat query_descriptors, Mat db_descriptors,
                                   const vector<KeyPoint>& query_keypoints,
                                   const vector<KeyPoint>& db_keypoints,
                                   Ptr<DescriptorMatcher> matcher,
                                   double ratio_threshold,
                                   bool use_distance_threshold) {
  map<int, list<DMatch>> best_match_per_trainidx;

  vector<vector<DMatch>> first_two_nns;

  if (ratio_threshold > 1.0) {
    //    // just return a bunch of stuff
    //    // for convenience, interpret the threshold as the num nearest
    //    neighbors
    //    matcher->knnMatch(query_descriptors, db_descriptors, first_two_nns,
    //                      static_cast<int>(ratio_threshold));
    //    for (vector<DMatch>& nn_matches : first_two_nns) {
    //      for (auto& match : nn_matches) {
    //        best_match_per_trainidx[match.trainIdx].push_back(match);
    //      }
    //    }

    // just return a bunch of stuff
    // for convenience, interpret the threshold as a threshold distance

    vector<vector<DMatch>> nns_in_radius;
    matcher->radiusMatch(query_descriptors, db_descriptors, nns_in_radius,
                         ratio_threshold);
    int num_matches = 0;
    for (vector<DMatch>& nn_matches : nns_in_radius) {
      for (auto& match : nn_matches) {
        ++num_matches;
        best_match_per_trainidx[match.trainIdx].push_back(match);
      }
    }
    cout << "found " << num_matches << " within radius " << ratio_threshold << endl;
  } else {
    matcher->knnMatch(query_descriptors, db_descriptors, first_two_nns, 2);

    for (vector<DMatch>& nn_matches : first_two_nns) {
      if (nn_matches.size() < 2) {
        continue;
      }
      if (nn_matches[0].distance < ratio_threshold * nn_matches[1].distance) {
        auto iter = best_match_per_trainidx.find(nn_matches[0].trainIdx);
        // could also do a ratio test in the other direction, but that
        // would require more bookkeeping
        if (iter == best_match_per_trainidx.end() ||
            iter->second.front().distance > nn_matches[0].distance) {
          // for the caffe descriptor, the margin for the hinge loss set to
          // 0.5. This makes a natural metric for detecting outliers.
          if (!use_distance_threshold || nn_matches[0].distance < 12.0) {
            best_match_per_trainidx[nn_matches[0].trainIdx].clear();
            best_match_per_trainidx[nn_matches[0].trainIdx].push_back(
                nn_matches[0]);
          }
        }
      }
    }

    // optional: also check the other direction
    if (false) {
      // for dense descriptors, it's likely that if a point A in image 1 has
      // nearest neighbor B in image 2, then the nearest neighbor of point B
      // won't be A--but it could be a point that's relatively close to A, ergo
      // it shoould be kept.
      matcher->knnMatch(db_descriptors, query_descriptors, first_two_nns, 1);
      for (vector<DMatch>& nn_matches : first_two_nns) {
        if (nn_matches.size() < 2) {
          continue;
        }
        // the names are reversed, since we're checking the opposite direction
        int testIdx1 = nn_matches[0].trainIdx;
        int trainIdx1 = nn_matches[0].queryIdx;
        float dist1 = nn_matches[0].distance;
//        int testIdx2 = nn_matches[1].trainIdx;
//        int trainIdx2 = nn_matches[1].queryIdx;
        float dist2 = nn_matches[1].distance;

        bool need_to_remove = false;
        if (dist1 > ratio_threshold * dist2) {
          // ratio test doesn't pass in the opposite direction
          need_to_remove = true;
        } else {
          // This doesn't seem to work--there are weird banding effects where
          // most descriptors in a band in an image are missing
          // Maybe that's due to reprojecting other keyframes? not sure.
//          if (best_match_per_trainidx.find(trainIdx1) !=
//                  best_match_per_trainidx.end() &&
//              best_match_per_trainidx[trainIdx1].front().queryIdx != testIdx1) {
//            // for dense descriptor: if the nearest neighbor isn't exactly the
//            // same, but it's in the same general area on the image plane, we're
//            // okay
//
//            int alternativeTestIdx =
//                best_match_per_trainidx[trainIdx1].front().queryIdx;
//            float dist =
//                cv::norm(query_keypoints[testIdx1].pt -
//                              query_keypoints[alternativeTestIdx].pt);
//            // this threshold should be a function of the size of the window
//            // used to compute the descriptor (scale for SIFT and receptive
//            // field for learned), but just guesstimate for now
//            if (dist > 30) {
//              cout << "orig: " << query_keypoints[testIdx1].pt
//                   << ", alt: " << query_keypoints[alternativeTestIdx].pt
//                   << ", norm: " << dist << endl;
//              // a different point is the nearest neighbor
//              need_to_remove = true;
//
//              cout << "removing (" << trainIdx1 << "," << testIdx1 << ") <==> ("
//                   << trainIdx1 << ","
//                   << best_match_per_trainidx[trainIdx1].front().queryIdx
//                   << ") because ids don't match" << endl;
//            }
//          }
        }

        if (need_to_remove) {
          best_match_per_trainidx.erase(trainIdx1);
        }
      }
    }
  }

  if (false) {
    // retain top percentile only
    vector<float> distances;
    for (auto& matchlist : best_match_per_trainidx) {
      for (auto& match : matchlist.second) {
        distances.push_back(match.distance);
      }
    }
    cout << "match count before percentile thresholding: " << distances.size()
         << endl;
    sort(distances.begin(), distances.end());
    float fraction_to_keep = 0.5;
    float threshold =
        distances[static_cast<int>(fraction_to_keep * distances.size())];
    for (auto match_map_iter = best_match_per_trainidx.begin();
         match_map_iter != best_match_per_trainidx.end();) {
      list<DMatch>& matchlist = match_map_iter->second;
      for (auto match_list_iter = matchlist.cbegin();
           match_list_iter != matchlist.cend();) {
        if (match_list_iter->distance > threshold) {
          matchlist.erase(match_list_iter++);
        } else {
          ++match_list_iter;
        }
      }
      if (matchlist.empty()) {
        best_match_per_trainidx.erase(match_map_iter++);
      } else {
        ++match_map_iter;
      }
    }
  }

  return best_match_per_trainidx;
}

}  // namespace sdl
