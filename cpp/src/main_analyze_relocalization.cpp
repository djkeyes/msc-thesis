#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <locale>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/operations.hpp"
#include "opencv2/core/eigen.hpp"

#include "Relocalization.h"
#include "FusedFeatureDescriptors.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace sdl;
using namespace cv;

bool use_second_best_for_debugging = false;
bool display_top_matching_images = false;
bool save_first_trajectory = true;
bool display_stereo_correspondences = true;

int vocabulary_size;
double epsilon_angle_deg;
double epsilon_translation;
string slam_method;

tuple<Mat, int, int> getDummyCalibration(Mat probe_image) {
	int img_width, img_height;
	img_width = probe_image.cols;
	img_height = probe_image.rows;

	Mat K = Mat::zeros(3, 3, CV_32FC1);
	K.at<float>(0, 0) = (img_width + img_height) / 2.;
	K.at<float>(1, 1) = (img_width + img_height) / 2.;
	K.at<float>(0, 2) = img_width / 2 - 0.5;
	K.at<float>(1, 2) = img_height / 2 - 0.5;
	K.at<float>(2, 2) = 1;
	return make_tuple(K, img_width, img_height);
}

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
class SceneParser {
public:
	virtual ~SceneParser() {
	}

	/*
	 * Parse a scene into a set of databases and a set of queries (which may be built from overlapping data).
	 */
	virtual void parseScene(vector<sdl::Database>& dbs, vector<sdl::Query>& queries) = 0;

	/*
	 * Given a frame, load the ground truth pose (as a rotation and translation matrix) from the dataset.
	 */
	virtual void loadGroundTruthPose(const sdl::Frame& frame, Mat& rotation, Mat& translation) = 0;
};
class SevenScenesParser: public SceneParser {
public:
	SevenScenesParser(const fs::path& directory) :
			directory(directory) {
	}
	virtual ~SevenScenesParser() {
	}
	virtual void parseScene(vector<sdl::Database>& dbs, vector<sdl::Query>& queries) {
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
				unique_ptr<Frame> frame(new sdl::Frame(id, cur_db.db_id));
				frame->setImageLoader([file]() {return imread(file.path().string());});
				frame->setPath(sequence_dir);
				if (!cache.empty()) {
					string image_filename(file.path().string());
					frame->setCachePath(cache / sequence_dir.filename());
					sorted_images.push_back(image_filename);
				}

				queries.emplace_back(cur_db.db_id, frame.get());
				cur_db.addFrame(move(frame));
			}

			sort(sorted_images.begin(), sorted_images.end());
			if(slam_method.find("DSO") == 0) {
				auto calib = getDummyCalibration(imread(sorted_images[0]));
				Mat K = get<0>(calib);
				int width = get<1>(calib);
				int height = get<2>(calib);
				cur_db.setMapper(new DsoMapGenerator(K, width, height, sorted_images, cur_db.getCachePath().string()));
			} else if (slam_method.length() > 0) {
				stringstream ss;
				ss << "SevenScenesParser for slam method '" << slam_method << "' not implemented!";
				throw runtime_error(ss.str());
			}

		}
	}

	virtual void loadGroundTruthPose(const sdl::Frame& frame, Mat& rotation, Mat& translation) {
		// frame i in db j -> seq-(j+1)/frame-XXXXXi.pose.txt
		stringstream ss;
		ss << "frame-" << setfill('0') << setw(6) << frame.index << ".pose.txt";
		fs::path pose_path = frame.framePath / ss.str();
		// operator+= is defined, but not operator+. Weird, eh?
		pose_path += ".pose.txt";

		ifstream ifs(pose_path.string());
		rotation.create(3, 3, CV_32FC1);
		translation.create(3, 1, CV_32FC1);
		// pose is stored as a 4x4 float matrix [R t; 0 1], so we only care about the first 3 rows.
		for (int i = 0; i < 3; i++) {
			ifs >> rotation.at<float>(i, 0);
			ifs >> rotation.at<float>(i, 1);
			ifs >> rotation.at<float>(i, 2);
			ifs >> translation.at<float>(i, 0);
		}

		ifs.close();
	}

	void setCache(fs::path cache_dir) {
		cache = cache_dir;
	}

private:
	fs::path directory;
	fs::path cache;
};

class TumParser: public SceneParser {
public:
	TumParser(const fs::path& directory) :
			directory(directory) {
	}
	virtual ~TumParser() {
	}
	virtual void parseScene(vector<sdl::Database>& dbs, vector<sdl::Query>& queries) {
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

		// each sequence can produce 1 database and several queries
		for (const fs::path& sequence_dir : sequences) {
			dbs.emplace_back();

			Database& cur_db = dbs.back();
			cur_db.db_id = dbs.size() - 1;
			if (!cache.empty()) {
				cur_db.setCachePath(cache / sequence_dir.filename());
			}

			// assumes all the images have been unzipped to a directory images/
			fs::path image_dir = sequence_dir / "images";
			vector<string> sorted_images;

			string calib = (sequence_dir / "camera.txt").string();
			string gamma_calib = (sequence_dir / "pcalib.txt").string();
			string vignette = (sequence_dir / "vignette.png").string();

			shared_ptr<ImageFolderReader> reader(
					new ImageFolderReader(image_dir.string(), calib, gamma_calib, vignette));
			for (auto file : fs::recursive_directory_iterator(image_dir)) {
				string name(file.path().filename().string());
				if (name.find(".jpg") == string::npos) {
					continue;
				}
				// these files are in the format XXXXX.jpg
				int id = stoi(name.substr(0, 5));
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
				if (!cache.empty()) {
					string image_filename(file.path().string());
					frame->setCachePath(cache / sequence_dir.filename());
					sorted_images.push_back(image_filename);
				}

				queries.emplace_back(cur_db.db_id, frame.get());
				cur_db.addFrame(move(frame));
			}

			sort(sorted_images.begin(), sorted_images.end());
			if(slam_method.find("DSO") == 0) {
				cur_db.setMapper(new DsoMapGenerator(sequence_dir.string()));
			} else if (slam_method.length() > 0) {
				stringstream ss;
				ss << "TumParser for slam method '" << slam_method << "' not implemented!";
				throw runtime_error(ss.str());
			}

			// preload the ground truth poses, so we don't have to seek every time
			// file is sequence_dir/groundtruthSync.txt
			fs::path pose_path = sequence_dir / "groundtruthSync.txt";
			ifstream pose_file(pose_path.string());

			string line;
			int index = 0;
			while(getline(pose_file, line)){
				// images/XXXXX.jpg -> line XXXXX of groundtruthSync.txt (both are 0-indexed)

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

				rotationsAndTranslationsByDatabaseAndFrame[cur_db.db_id][index] = make_tuple(R, t);
				++index;
			}
			pose_file.close();
		}

	}

	virtual void loadGroundTruthPose(const sdl::Frame& frame, Mat& rotation, Mat& translation) {
		// for the TUM dataset, initial and final ground truth is available (I
		// assume during this time, the camera is tracked by an multi-view IR
		// tracker system or something), but otherwise the pose is recorded as NaN

		// It would almost certainly be better to load all the poses at once,
		// or at least to cache filehandles instead of re-opening the file for
		// each one. If/ opening/seeking/closing becomes a bottleneck, refactor
		// this.

		rotation = get<0>(rotationsAndTranslationsByDatabaseAndFrame.at(frame.dbId).at(frame.index));
		translation = get<1>(rotationsAndTranslationsByDatabaseAndFrame.at(frame.dbId).at(frame.index));
	}

	void setCache(fs::path cache_dir) {
		cache = cache_dir;
	}

private:
	fs::path directory;
	fs::path cache;
	map<int, map<int, tuple<Mat, Mat>>> rotationsAndTranslationsByDatabaseAndFrame;
};

unique_ptr<SceneParser> scene_parser;


class MatchingMethod {
public:
	virtual ~MatchingMethod() = default;

	virtual bool needs3dDatabasePoints() = 0;
	virtual bool needs3dQueryPoints() = 0;

	void doMatching(const Query& q, const Result& top_result, bool same_database, const vector<Point2f>& query_pts,
			const vector<Point2f>& database_pts) {
		Mat R, t, inlier_mask;
		internalDoMatching(top_result, query_pts, database_pts, inlier_mask, R, t);

		updateResults(q, top_result, same_database, inlier_mask, R, t, query_pts, database_pts);
	}

	void setK(Mat K_) {
		K = K_;
	}

	void updateResults(const Query& q, const Result& top_result, bool same_database, Mat inlier_mask, Mat R, Mat t,
			const vector<Point2f>& query_pts, const vector<Point2f>& database_pts) {

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
			ss << "Stereo correspondences for query " << q.getFrame()->index << "(db " << q.getFrame()->dbId << ") <==> result "
					<< top_result.frame.index << " (db " << top_result.frame.dbId << ")";
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

		// Note: iff x = nan, then x != x is true.
		// Use this to detect nan-filled matrices.
		if ((countNonZero(db_t_gt != db_t_gt) > 0) || countNonZero(query_t_gt != query_t_gt)
				|| (countNonZero(db_R_gt != db_R_gt) > 0) || countNonZero(query_R_gt != query_R_gt)) {
			// For the TUM dataset, poses are only known at the beginning and
			// end, so nans are reasonable. For other datasets, this should not
			// occur. If we wanted to compare to full trajectories, we could
			// compute our own SFM pipeline, although that biases our result.
			return;
		}

		// This is only known up to scale. So cheat by fixing the scale to the ground truth.
		if (!needs3dDatabasePoints() && !needs3dQueryPoints()) {
			t *= norm(db_t_gt - query_t_gt) / norm(t);
		} else {
			// Even if we're matching 3D points, the scale of the ground truth
			// and scale of the reconstruction might not be the same. So
			// rescale to address that.
			// TODO: just rescale the reconstruction once at the beginning.
			// TODO: use the reconstruction as the ground truth, so no need for rescaling.
			// TODO: what happens if t is very close to 0? or if db_t_gt == query_t_gt?
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
		// if v is the unit vector in direction z, we want acos(v dot Rv) = acos(R_22)
		float angle_error_rad = acos(rotation_diff.at<double>(2, 2));
		float angle_error_deg = (180. * angle_error_rad / M_PI);

		if ((translation_error < epsilon_translation) && (angle_error_deg < epsilon_angle_deg)) {
			if (same_database) {
				epsilon_accurate_train_queries++;
			} else {
				epsilon_accurate_test_queries++;
			}
		}

		if (save_first_trajectory && q.getParentDatabaseId() == 0 && top_result.frame.dbId == 0) {
			actualAndExpectedPoses.insert(make_pair(q.getFrame()->index, make_pair(estimated_t, query_t_gt)));
		}
	}

	void printResults(int total_train_queries, int total_test_queries) {
		double train_reg_accuracy = static_cast<double>(high_inlier_train_queries) / total_train_queries;
		double test_reg_accuracy = static_cast<double>(high_inlier_test_queries) / total_test_queries;
		double train_loc_accuracy = static_cast<double>(epsilon_accurate_train_queries) / total_train_queries;
		double test_loc_accuracy = static_cast<double>(epsilon_accurate_test_queries) / total_test_queries;

		cout << endl;
		cout << "Processed " << (total_test_queries + total_train_queries) << " queries! (" << total_train_queries
				<< " queries on their own train set database, and " << total_test_queries
				<< " on separate test set databases)" << endl;
		cout << "Train set registration accuracy (num inliers after pose recovery >= 12): " << train_reg_accuracy
				<< endl;
		cout << "Test set registration accuracy (num inliers after pose recovery >= 12): " << test_reg_accuracy << endl;
		cout << "Train set localization accuracy (error < " << epsilon_translation << ", " << epsilon_angle_deg
				<< " deg): " << train_loc_accuracy << endl;
		cout << "Test set localization accuracy (error < " << epsilon_translation << ", " << epsilon_angle_deg
				<< " deg): " << test_loc_accuracy << endl;

		if (save_first_trajectory) {
			ofstream actual_file("actual.txt");
			ofstream expected_file("expected.txt");
			for (const auto& element : actualAndExpectedPoses) {
				const Mat& actual = element.second.first;
				const Mat& expected = element.second.second;
				actual_file << actual.at<float>(0, 0) << " " << actual.at<float>(1, 0) << " "
						<< actual.at<float>(2, 0) << endl;
				expected_file << expected.at<float>(0, 0) << " " << expected.at<float>(1, 0) << " "
						<< expected.at<float>(2, 0) << endl;
			}
			actual_file.close();
			expected_file.close();
		}

	}

	unsigned int epsilon_accurate_test_queries = 0;
	unsigned int high_inlier_test_queries = 0;
	unsigned int epsilon_accurate_train_queries = 0;
	unsigned int high_inlier_train_queries = 0;

protected:
	virtual void internalDoMatching(const Result& top_result, const vector<Point2f>& query_pts,
			const vector<Point2f>& database_pts, Mat& inlier_mask, Mat& R, Mat& t) = 0;

	Mat K;

private:
	map<int, pair<Mat, Mat>> actualAndExpectedPoses;
};

// Match between images and compute homography
class Match_2d_2d_5point: public MatchingMethod {

	bool needs3dDatabasePoints() override {
		return false;
	}
	bool needs3dQueryPoints() override {
		return false;
	}

	void internalDoMatching(const Result& top_result, const vector<Point2f>& query_pts, const vector<Point2f>& database_pts, Mat& inlier_mask,
			Mat& R, Mat& t) override {
		int ransac_threshold = 3; // in pixels, for the sampson error
		// TODO: the number of inliers registered for trainset images is
		// extremely low. Perhaps they are coplanar or otherwise
		// ill-conditioned? Setting confidence to 1 forces RANSAC to use
		// its default maximum number of iterations (1000), but it would
		// be better to filter the data, or increase the max number of
		// trials.
		double confidence = 0.999999;

		// pretty sure this is the correct order
		Mat E = findEssentialMat(database_pts, query_pts, K, RANSAC, confidence, ransac_threshold, inlier_mask);
		recoverPose(E, database_pts, query_pts, K, R, t, inlier_mask);
		// watch out: E, R, & t are double precision (even though we only ever passed floats)

		// Convert from project matrix parameters (world to camera) to camera pose (camera to world)
		R = R.t();
		t = -R * t;
	}
};
// Match 2D to 3D using image retrieval as a proxy for possible co-visibility,
// and use depth values (from slam) to recover transformation
class Match_2d_3d_dlt: public MatchingMethod {

	bool needs3dDatabasePoints() override {
		return true;
	}
	bool needs3dQueryPoints() override {
		return false;
	}

	void internalDoMatching(const Result& top_result, const vector<Point2f>& query_pts, const vector<Point2f>& database_pts, Mat& inlier_mask,
			Mat& R, Mat& t) override {

		int num_iters = 1000;
		double confidence = 0.999999;
		double ransac_threshold = 8.0;

		// lookup scene coords from database_pts
		SparseMat scene_coords = top_result.frame.loadSceneCoordinates();
		vector<Point3f> scene_coord_vec;
		scene_coord_vec.reserve(database_pts.size());
		for (const auto& point : database_pts) {
			scene_coord_vec.push_back(
					scene_coords.value<Point3f>(static_cast<int>(point.y), static_cast<int>(point.x)));
		}

		Mat rvec, inliers;

		solvePnPRansac(scene_coord_vec, query_pts, K, noArray(), rvec, t, false, num_iters, ransac_threshold, confidence,
				inliers);
		Rodrigues(rvec, R);

		R.convertTo(R, CV_32F);
		t.convertTo(t, CV_32F);

		// Convert from project matrix parameters (world to camera) to camera pose (camera to world)
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
	po::options_description commandline_exclusive("Allowed options from terminal");
	commandline_exclusive.add_options()
			("help", "Print this help message.")
			("config", po::value<string>(),	"Path to a config file, which can specify any other argument.");

	po::options_description general_args("Allowed options from terminal or config file");
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
			("mapping_method", po::value<string>()->default_value(""), "Mapping/SLAM method for building a map"
					" to relocalize against. Can be 'DSO' or left empty.")
			("pose_estimation_method", po::value<string>()->default_value(""), "Mapping/SLAM method for building a map"
					" to relocalize against. Can be '5point_E' to decompose an essential matrix (and use the ground-truth scale),"
					" 'DLT' to compute a direct linear transform (requires mapping_method to be specified), or left empty.");

	po::options_description commandline_args;
	commandline_args.add(commandline_exclusive).add(general_args);

	// check for config file
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(commandline_exclusive).allow_unregistered().run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		// print help and exit
		usage(argv, commandline_args);
		exit(0);
	}

	if (vm.count("config")) {
		ifstream ifs(vm["config"].as<string>());
		if (!ifs) {
			cout << "could not open config file " << vm["config"].as<string>() << endl;
			exit(1);
		}
		vm.clear();
		// since config is added last, commandline args have precedence over args in the config file.
		po::store(po::command_line_parser(argc, argv).options(commandline_args).run(), vm);
		po::store(po::parse_config_file(ifs, general_args), vm);
	} else {
		vm.clear();
		po::store(po::command_line_parser(argc, argv).options(commandline_args).run(), vm);
	}
	po::notify(vm);

	fs::path cache_dir;
	if (vm.count("cache")) {
		cache_dir = fs::path(vm["cache"].as<string>());
	}

	vocabulary_size = vm["vocabulary_size"].as<int>();
	epsilon_angle_deg = vm["epsilon_angle_deg"].as<double>();
	epsilon_translation = vm["epsilon_translation"].as<double>();
	slam_method = vm["mapping_method"].as<string>();
	string matching_method_str = vm["pose_estimation_method"].as<string>();
	if (matching_method_str.length() == 0 || matching_method_str.find("5point_E") == 0) {
		matching_method = unique_ptr<MatchingMethod>(new Match_2d_2d_5point());
	} else if (matching_method_str.find("DLT") == 0) {
		matching_method = unique_ptr<MatchingMethod>(new Match_2d_3d_dlt());
	} else {
		throw runtime_error("Invalid value for pose_estimation_method!");
	}

	if (slam_method.length() == 0 && matching_method_str.find("DLT") == 0) {
		throw runtime_error("A 3D matching method was specified, but no SLAM method was specified.");
	}

	if (vm.count("scene")) {

		string scene_type(vm["scene"].as<string>());
		// currently only one supported scene
		if (scene_type.find("7scenes") == 0) {
			fs::path directory(vm["datadir"].as<string>());
			SevenScenesParser* parser = new SevenScenesParser(directory);
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
int main(int argc, char** argv) {

	parseArguments(argc, argv);

	vector<sdl::Database> dbs;
	vector<sdl::Query> queries;

	scene_parser->parseScene(dbs, queries);

	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(20000, 3, 0.005, 80);

	Ptr<Feature2D> detector;
	if(matching_method->needs3dDatabasePoints()){
		detector = Ptr<Feature2D>(new NearestDescriptorAssigner(*sift));
	} else {
		detector = sift;
	}

	for (sdl::Database& db : dbs) {
		db.setVocabularySize(vocabulary_size);
		db.setupFeatureDetector(matching_method->needs3dDatabasePoints());
		db.setDescriptorExtractor(detector);
		db.setBowExtractor(makePtr<BOWSparseImgDescriptorExtractor>(sift, FlannBasedMatcher::create()));

		db.train();
	}
	for (sdl::Query& query : queries) {
		// TODO: incorporate previous frames into each query (or maybe just use
		// the cached depth map, which would be fine for cross-database queries)
		query.setupFeatureDetector(matching_method->needs3dQueryPoints());
		query.setDescriptorExtractor(sift);
		query.computeFeatures();
	}

	// Ideally most queries should be valid. :(
	// If this is nonzero, it's probably due to a lack of keyframe density
	// during visual odometry, and discarding empty frames isn't so bad.
	int orig_query_count = queries.size();
	queries.erase(std::remove_if(queries.begin(), queries.end(), [](Query& q) {return q.getDescriptors().empty();}),
			queries.end());
	int pruned_query_count = queries.size();
	cout << "Have " << pruned_query_count << " valid queries after removing " << (orig_query_count - pruned_query_count)
			<< " empty queries." << endl;

	// TODO: use an actually calibrated camera model
	// supposedly COLMAP can estimate calibration from video as part of its optimization?
	Mat K = get<0>(getDummyCalibration(queries[0].getFrame()->imageLoader()));
	matching_method->setK(K);

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
			if (total_so_far % 100 == 0 || total_so_far == total) {
				cout << "\r" << fixed << setprecision(4) << static_cast<double>(total_so_far) / total * 100. << "% ("
						<< total_so_far << "/" << total << ")" << flush;
			}

			vector<sdl::Result> results = dbs[i].lookup(queries[j], num_to_return);
			sdl::Result& top_result = results[use_second_best_for_debugging ? 1 : 0];

			// display the result in a pretty window
			if (j < 5 && display_top_matching_images) {

				int width = 640, height = 640;
				int rows = 3, cols = 3;
				int bordersize = 3;
				Mat grid(height, width, CV_8UC3);
				rectangle(grid, Point(0, 0), Point(width, height), Scalar(255, 255, 255), FILLED, LINE_8);
				rectangle(grid, Point(0, 0), Point(width / cols, height / rows), Scalar(0, 0, 255), FILLED, LINE_8);

				for (int c = 0; c < cols; ++c) {
					for (int r = 0; r < rows; ++r) {
						Point start(bordersize + c * width / cols, bordersize + r * height / rows);
						Point end((c + 1) * width / cols - bordersize, (r + 1) * height / rows - bordersize);
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
						putText(image, text, Point(tilewidth / 2 - 30, 15), FONT_HERSHEY_PLAIN, 0.9,
								Scalar(255, 255, 255));
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
						Point2f to = result_keypoints[correspondence.trainIdx].pt + Point2f(width / 2, 0);

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

			matching_method->doMatching(queries[j], top_result, queries[j].getParentDatabaseId() == dbs[i].db_id,
					query_pts, database_pts);
		}

	}

	matching_method->printResults(total_train_queries, total_test_queries);

	return 0;
}
