
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/QR"
#include "Eigen/StdVector"

#include "geometricburstiness/inverted_index.h"
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
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].angle), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&keypoints[i].response), sizeof(float));
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
void Frame::saveSceneCoordinates(cv::SparseMat coordinate_map) const {
  // Can't use imwrite/read, since that only works on cv::mat of type CV_8BC3

  assert(coordinate_map.type() == CV_32FC3);

  fs::path filename(getSceneCoordinateFilename().string());
  fs::create_directories(filename.parent_path());

  ofstream ofs(filename.string(), ios_base::out | ios_base::binary);

  uint32_t rows = coordinate_map.size(0);
  uint32_t cols = coordinate_map.size(1);
  uint32_t size = coordinate_map.nzcount();
  ofs.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
  ofs.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
  for (auto iter = coordinate_map.begin(); iter != coordinate_map.end();
       ++iter) {
    cv::Vec3f& val = iter.value<cv::Vec3f>();
    uint16_t row = iter.node()->idx[0];
    uint16_t col = iter.node()->idx[1];
    ofs.write(reinterpret_cast<const char*>(&row), sizeof(uint16_t));
    ofs.write(reinterpret_cast<const char*>(&col), sizeof(uint16_t));
    ofs.write(reinterpret_cast<const char*>(&val), sizeof(cv::Vec3f));
  }

  ofs.close();
}
cv::SparseMat Frame::loadSceneCoordinates() const {
  ifstream ifs(getSceneCoordinateFilename().string(),
               ios_base::in | ios_base::binary);

  uint32_t rows, cols, size;
  ifs.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
  ifs.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
  cv::SparseMat coordinate_map;
  int dims[] = {static_cast<int>(rows), static_cast<int>(cols)};
  coordinate_map.create(2, dims, CV_32FC3);
  for (int i = 0; i < static_cast<int>(size); ++i) {
    uint16_t row, col;
    ifs.read(reinterpret_cast<char*>(&row), sizeof(uint16_t));
    ifs.read(reinterpret_cast<char*>(&col), sizeof(uint16_t));
    cv::Vec3f val;
    ifs.read(reinterpret_cast<char*>(&val), sizeof(cv::Vec3f));
    coordinate_map.ref<cv::Vec3f>(static_cast<int>(row),
                                  static_cast<int>(col)) = val;
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
  ifstream ifs(getPoseFilename().string(),
               ios_base::in | ios_base::binary);

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
      associateWithDepthMaps(false),
      frames(new map<int, unique_ptr<Frame>>()),
      pInvertedIndexImpl(new Database::InvertedIndexImpl()) {}
// Need to explicitly define destructor and move constructor, otherwise
// compiler can't handle unique_ptrs with forward-declared types.
Database::Database(Database&&) = default;
Database::~Database() = default;

vector<Result> Database::lookup(Query& query, unsigned int num_to_return) {
  vector<int> assignments;
  const Mat descriptors = query.computeDescriptors();
  bowExtractor->computeAssignments(descriptors, assignments);

  const vector<KeyPoint>& keypoints = query.getKeypoints();
  int num_features = keypoints.size();

  int descriptor_size = descriptorExtractor->descriptorSize();

  vector<geometric_burstiness::QueryDescriptor<64>> query_descriptors(
      num_features);

  for (int j = 0; j < num_features; ++j) {
    Map<MatrixXf> descriptor(reinterpret_cast<float*>(descriptors.row(j).data),
                             descriptor_size, 1);

    pInvertedIndexImpl->invertedIndex.PrepareQueryDescriptor(
        descriptor, &(query_descriptors[j]));
    // TODO: use more than 1 nearest word?
    int nearest_word = assignments[j];
    query_descriptors[j].relevant_word_ids.push_back(nearest_word);
    query_descriptors[j].x = keypoints[j].pt.x;
    query_descriptors[j].y = keypoints[j].pt.y;
    // TODO
    query_descriptors[j].a = 0;
    query_descriptors[j].b = 0;
    query_descriptors[j].c = 0;
    query_descriptors[j].feature_id = j;
  }
  for (int j = 0; j < num_features; ++j) {
    for (unsigned int k = 0; k < query_descriptors[j].relevant_word_ids.size();
         k++) {
      query_descriptors[j].max_hamming_distance_per_word.push_back(32);
    }
  }

  vector<geometric_burstiness::ImageScore> image_scores;
  pInvertedIndexImpl->invertedIndex.QueryIndex(query_descriptors,
                                               &image_scores);

  vector<Result> results;
  for (unsigned int i = 0;
       i < min(static_cast<unsigned int>(image_scores.size()), num_to_return);
       ++i) {
    results.emplace_back(*frames->at(image_scores[i].image_id));
    // discard some of the worse correspondences
    vector<float> weights;
    for (const auto& correspondence : image_scores[i].matches) {
      weights.push_back(correspondence.weight);
    }
    std::sort(weights.begin(), weights.end());
    float cutoff =
        weights[max(0, min(static_cast<int>(weights.size() - 12),
                           static_cast<int>(weights.size() * 0.25)))];

    for (const auto& correspondence : image_scores[i].matches) {
      int db_feature_id =
          pInvertedIndexImpl->invertedIndex
              .GetIthEntryForWord(correspondence.db_feature_word,
                                  correspondence.db_feature_index)
              .feature_id;

      if (correspondence.weight < cutoff) {
        continue;
      }
      results.back().matches.emplace_back(
          correspondence.query_feature_id, db_feature_id,
          image_scores[i].image_id, correspondence.weight);
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

void Database::doMapping() {
  if (!needToRecomputeSceneCoordinates()) {
    return;
  }
  // first run the full mapping algorithm
  mapGen->runVisualOdometry();

  // then fetch the depth maps / poses, and convert them to 2D -> 3D image maps
  map<int, cv::SparseMat> scene_coordinate_maps =
      mapGen->getSceneCoordinateMaps();
  std::map<int, dso::SE3>* poses = mapGen->getPoses();

  const int* dims = scene_coordinate_maps.begin()->second.size();
  for (unsigned int frame_id = 0; frame_id < frames->size(); ++frame_id) {
    auto iter = scene_coordinate_maps.find(frame_id);
    if (iter == scene_coordinate_maps.end()) {
      cv::SparseMat empty;
      empty.create(2, dims, CV_32FC3);
      frames->at(frame_id)->saveSceneCoordinates(empty);
    } else {
      frames->at(frame_id)->saveSceneCoordinates(iter->second);
    }

    auto pose_iter = poses->find(frame_id);
    if (pose_iter == poses->end()) {
      double nan = numeric_limits<double>::quiet_NaN();
      dso::SE3 invalid(dso::SE3(Eigen::Quaterniond::Identity(),
                                dso::SE3::Point(nan, nan, nan)));
      frames->at(frame_id)->savePose(invalid);
    } else {
      frames->at(frame_id)->savePose(pose_iter->second);
    }
  }

  // Also save a pointcloud, for debugging
  mapGen->savePointCloudAsPcd((cachePath / "pointcloud.pcd").string());
  mapGen->savePointCloudAsPly((cachePath / "pointcloud.ply").string());

  // clear mapGen, since it hogs a bunch of memory
  mapGen.reset();
}
bool Database::needToRecomputeSceneCoordinates() const {
  unsigned int num_saved_coords = 0;
  for (auto file : fs::recursive_directory_iterator(cachePath)) {
    if (file.path().filename().string().find("scene_coords") != string::npos) {
      num_saved_coords++;
    }
  }
  return num_saved_coords != frames->size();
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
        SparseMat scene_coords = frame->loadSceneCoordinates();

        with_depth->setCurrentSceneCoords(scene_coords);
      }
      descriptorExtractor->detect(colorImage, image_keypoints[frame->index]);
      descriptorExtractor->compute(colorImage, image_keypoints[frame->index],
                                   image_descriptors[frame->index]);

      if (display_keypoints && num_displayed < 5 &&
          !image_keypoints[frame->index].empty()) {
        drawKeypoints(colorImage, image_keypoints[frame->index], colorImage,
                      Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
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
  }
  return total_descriptors;
}
void Database::doClustering(const map<int, Mat>& image_descriptors) {
  if (!loadVocabulary(vocabulary)) {
    int max_iters = 10;
    TermCriteria terminate_criterion(TermCriteria::MAX_ITER, max_iters, 0.0);
    BOWApproxKMeansTrainer bow_trainer(vocabulary_size, terminate_criterion);

    for (const auto& element : *frames) {
      int index = element.second->index;
      if (!image_descriptors.at(index).empty()) {
        int descriptor_count = image_descriptors.at(index).rows;

        for (int i = 0; i < descriptor_count; i++) {
          bow_trainer.add(image_descriptors.at(index).row(i));
        }
      }
    }
    vocabulary = bow_trainer.cluster();
    saveVocabulary(vocabulary);
  }
  bowExtractor->setVocabulary(vocabulary);
}

map<int, vector<int>> Database::computeBowDescriptors(
    const map<int, Mat>& image_descriptors) {
  // Create training data by converting each keyframe to a bag of words
  map<int, vector<int>> assignments;
  for (const auto& element : *frames) {
    int index = element.second->index;

    bowExtractor->computeAssignments(image_descriptors.at(index),
                                     assignments[index]);
  }
  return assignments;
}

MatrixXf Database::generateRandomProjection(int descriptor_size, int num_rows) {
  default_random_engine generator;
  normal_distribution<float> distribution(0.0, 1.0);

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
    const MatrixXf& projection_matrix, const map<int, Mat>& image_descriptors,
    const map<int, vector<int>> descriptor_assignments) {
  Matrix<float, 64, Eigen::Dynamic> hamming_thresholds;
  hamming_thresholds.resize(64, vocabulary_size);
  int descriptor_size = descriptorExtractor->descriptorSize();

  int num_images = static_cast<int>(image_descriptors.size());
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

  vector<int> randomly_permuted_db_ids(num_images);
  for (int i = 0; i < num_images; ++i) {
    randomly_permuted_db_ids[i] = i;
  }
  random_shuffle(randomly_permuted_db_ids.begin(),
                 randomly_permuted_db_ids.end());

  cout << " Determining relevant images per word " << endl;
  const int kNumDesiredDesc = 10000;
  for (int i = 0; i < num_images; ++i) {
    if (num_missing_words == 0) break;

    int id = randomly_permuted_db_ids[i];

    const Mat descriptors = image_descriptors.at(id);

    int num_features = descriptors.rows;

    if (num_features == 0) {
      continue;
    }

    for (int j = 0; j < num_features; ++j) {
      const int closest_word = descriptor_assignments.at(id)[j];
      if (num_desc_per_word[closest_word] >= kNumDesiredDesc) {
        continue;
      }

      // map the opencv memory to an eigen matrix
      // Ostensibly the opencv matrix is 1xD, and the eigen matrix needs
      // to be Dx1, but it doesn't really matter since it's 1-dimensional
      // and stored contiguously.
      Map<MatrixXf> descriptor(reinterpret_cast<float*>(descriptors.row(j).data),
                               descriptor_size, 1);

      Eigen::Matrix<float, 64, 1> proj_sift = projection_matrix * descriptor;

      for (int k = 0; k < 64; ++k) {
        entries_per_word[closest_word][k].push_back(proj_sift[k]);
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
    const map<int, vector<KeyPoint>>& image_keypoints,
    const map<int, Mat>& image_descriptors) {

  cout << "Computing bow descriptors for each image in training set using "
          "nearest neighbor to each descriptor..."
       << endl;
  map<int, vector<int>> descriptor_assignments =
      computeBowDescriptors(image_descriptors);
  cout << "Finished computing bags of words." << endl;

  MatrixXf projection_matrix =
      generateRandomProjection(descriptorExtractor->descriptorSize(), 64);
  MatrixXf hamming_thresholds = computeHammingThresholds(
      projection_matrix, image_descriptors, descriptor_assignments);

  geometric_burstiness::InvertedIndex<64>& inverted_index =
      pInvertedIndexImpl->invertedIndex;
  cout << "initializing index to vocabulary size " << vocabulary_size << endl;
  inverted_index.InitializeIndex(vocabulary_size);
  cout << "finished initialization" << endl;

  inverted_index.SetProjectionMatrix(projection_matrix);
  inverted_index.SetHammingThresholds(hamming_thresholds);

  int descriptor_size = descriptorExtractor->descriptorSize();

  int total_number_entries = 0;
  for (const auto& element : *frames) {
    int index = element.second->index;
    const vector<cv::KeyPoint>& keypoints = image_keypoints.at(index);
    const Mat& descriptors = image_descriptors.at(index);

    int num_features = static_cast<int>(keypoints.size());

    if (num_features == 0) {
      continue;
    }

    for (int j = 0; j < num_features; ++j, ++total_number_entries) {
      geometric_burstiness::InvertedFileEntry<64> entry;
      entry.image_id = index;
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

      const int closest_word = descriptor_assignments.at(index)[j];
      if (closest_word < 0 || closest_word >= vocabulary_size) {
        throw runtime_error("Impossible word " + closest_word);
      }

      Map<MatrixXf> descriptor(reinterpret_cast<float*>(descriptors.row(j).data),
                               descriptor_size, 1);

      inverted_index.AddEntry(entry, closest_word, descriptor);
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

  saveInvertedIndex(*pInvertedIndexImpl);
}
void Database::train() {
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

  bool has_index = loadInvertedIndex(*pInvertedIndexImpl);

  // check if we haven't compute the descriptors yet, or if we have but still
  // need to cluster the vocabulary, or we need to compute word assignments for
  // the inverted index
  if (!hasCachedDescriptors() || !hasCachedVocabularyFile() || !has_index) {
    cout << "computing descriptors for each keyframe..." << endl;
    int descriptor_count =
        computeDescriptorsForEachFrame(image_keypoints, image_descriptors);
    cout << "computed " << descriptor_count << " descriptors in "
         << frames->size() << " frames." << endl;
  } else {
    cout << "Already have cached descriptors." << endl;
  }

  // even if the vocab is already cached, we need to load it so we can find NNs
  // for queriesx) {
  cout << "Training vocabulary..." << endl;
  doClustering(image_descriptors);
  cout << "Finished training vocabulary." << endl;

  if (!has_index) {
    cout << "Building inverted index..." << endl;
    buildInvertedIndex(image_keypoints, image_descriptors);
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

}  // namespace sdl
