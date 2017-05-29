#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/operations.hpp"

#include "Relocalization.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace sdl;
using namespace cv;

bool display_top_matching_images = false;

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

			for (auto file : fs::recursive_directory_iterator(sequence_dir)) {
				string name(file.path().filename().string());
				if (name.find(".color.png") == string::npos) {
					continue;
				}
				// these files are in the format frame-XXXXXX.color.png
				int id = stoi(name.substr(6, 6));
				unique_ptr<Frame> frame(new sdl::Frame(id));
				frame->setImagePath(file);
				if (!cache.empty()) {
					frame->setCachePath(cache / sequence_dir.filename());
				}

				Query q(cur_db.db_id, frame.get());
				queries.push_back(q);
				cur_db.addFrame(move(frame));
			}
		}
	}

	virtual void loadGroundTruthPose(const sdl::Frame& frame, Mat& rotation, Mat& translation) {
		// frame-XXXXXX.color.png -> frame-XXXXXX.pose.txt
		fs::path pose_path = frame.imagePath.parent_path() / frame.imagePath.stem().stem();
		// operator+= is defined, but not operator+. Weird, eh?
		pose_path += ".pose.txt";

		ifstream ifs(pose_path.string());
		rotation.create(3, 3, CV_64FC1);
		translation.create(3, 1, CV_64FC1);
		// pose is stored as a 4x4 float matrix [R t; 0 1], so we only care about the first 3 rows.
		for (int i = 0; i < 3; i++) {
			ifs >> rotation.at<double>(i, 0);
			ifs >> rotation.at<double>(i, 1);
			ifs >> rotation.at<double>(i, 2);
			ifs >> translation.at<double>(i, 0);
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

unique_ptr<SceneParser> scene_parser;
int vocabulary_size;
double epsilon_angle_deg;
double epsilon_translation;

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
			("epsilon_angle_deg", po::value<double>(), "Angle in degrees for a pose to be considered accurate.")
			("epsilon_translation", po::value<double>(), "Distance in the scene coordinate system for a pose"
					" to be considered accurate.");

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

	string datadir(argv[1]);
	vector<sdl::Database> dbs;
	vector<sdl::Query> queries;

	scene_parser->parseScene(dbs, queries);

	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();

	for (auto& db : dbs) {
		db.setVocabularySize(vocabulary_size);
		db.setFeatureDetector(sift);
		db.setDescriptorExtractor(sift);
		db.setBowExtractor(makePtr<BOWSparseImgDescriptorExtractor>(sift, FlannBasedMatcher::create()));
		db.train();
	}
	for (sdl::Query& query : queries) {
		query.setFeatureDetector(sift);
		query.setDescriptorExtractor(sift);
		query.computeFeatures();
	}

	int img_width, img_height;
	{
		Mat probe_image = queries[0].readColorImage();
		img_width = probe_image.cols;
		img_height = probe_image.rows;
	}
	int num_to_return = 8;

	unsigned int total_test_queries = 0;
	unsigned int epsilon_accurate_test_queries = 0;
	unsigned int total_train_queries = 0;
	unsigned int epsilon_accurate_train_queries = 0;
	cout << endl;

	map<int, pair<Mat, Mat>> actualAndExpectedPoses;

	for (unsigned int i = 0; i < dbs.size(); i++) {
		for (unsigned int j = 0; j < queries.size(); j++) {
			if (queries[j].parent_database_id == dbs[i].db_id) {
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
			sdl::Result& top_result = results[0];

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
							image = imread(results[index].frame.imagePath.string());
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

					Mat db_image = imread(top_result.frame.imagePath.string());
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
			int ransac_threshold = 5; // in pixels
			double confidence = 0.999;

			// TODO: use an actually calibrated camera model
			Mat K = Mat::zeros(3, 3, CV_32FC1);
			K.at<float>(0, 0) = 1;//585;
			K.at<float>(1, 1) = 1;//585;
			K.at<float>(0, 2) = img_width / 2 - 0.5;
			K.at<float>(1, 2) = img_height / 2 - 0.5;
			K.at<float>(2, 2) = 1;

			Mat E, R, t;
			// pretty sure this is the correct order
			E = findEssentialMat(database_pts, query_pts, K, RANSAC, confidence, ransac_threshold);
			recoverPose(E, database_pts, query_pts, K, R, t);
			// watch out: E, R, & t are double precision (even though we only ever passed floats)

			Mat db_R_gt, db_t_gt;
			scene_parser->loadGroundTruthPose(top_result.frame, db_R_gt, db_t_gt);
			Mat query_R_gt, query_t_gt;
			scene_parser->loadGroundTruthPose(*queries[j].frame, query_R_gt, query_t_gt);
			Mat estimated_R = db_R_gt * R;
			Mat estimated_t = db_R_gt * t + db_t_gt;

			double translation_error = norm(query_t_gt, estimated_t);
			// compute angle between rotation matrices
			Mat rotation_diff = (query_R_gt * estimated_R.t());
			// if v is the unit vector in direction x, we want acos(v dot Rv) = acos(R_00)
			float angle_error_rad = acos(rotation_diff.at<double>(0, 0));
			float angle_error_deg = (180. * angle_error_rad / M_PI);

			if ((translation_error < epsilon_translation) && (angle_error_deg < epsilon_angle_deg)) {
				if (queries[j].parent_database_id == dbs[i].db_id) {
					epsilon_accurate_train_queries++;
				} else {
					epsilon_accurate_test_queries++;
				}
			}

			if ((i == 0) && (queries[j].parent_database_id != dbs[i].db_id)) {
				cout << "query is " <<  queries[j].frame->imagePath.string() << endl;
				cout << "db image is " << top_result.frame.imagePath.string() << endl;
				cout << "translation error: " << translation_error << endl;
				cout << "estimated translation: " << estimated_t.t() << endl;
				cout << "db translation: " << db_t_gt.t() << endl;
				cout << "query translation: " << query_t_gt.t() << endl;
				cout << "database " << i << ", query with database "
						<< queries[j].parent_database_id << endl;
				cout << endl;
				actualAndExpectedPoses.insert(make_pair(queries[j].frame->index, make_pair(estimated_t, query_t_gt)));
			}
		}

	}

	ofstream actual_file("actual.txt");
	ofstream expected_file("expected.txt");
	for (const auto& element : actualAndExpectedPoses) {
		const Mat& actual = element.second.first;
		const Mat& expected = element.second.second;
		actual_file << actual.at<double>(0, 0) << " " << actual.at<double>(1, 0) << " " << actual.at<double>(2, 0)
				<< endl;
		expected_file << expected.at<double>(0, 0) << " " << expected.at<double>(1, 0) << " "
				<< expected.at<double>(2, 0) << endl;
	}
	actual_file.close();
	expected_file.close();


	double train_accuracy = static_cast<double>(epsilon_accurate_train_queries) / total_train_queries;
	double test_accuracy = static_cast<double>(epsilon_accurate_test_queries) / total_test_queries;

	cout << endl;
	cout << "Processed " << (total_test_queries + total_train_queries) << " queries! (" << total_train_queries
			<< " queries on their own train set database, and " << total_test_queries
			<< " on separate test set databases)" << endl;
	cout << "Train set accuracy (error < " << epsilon_translation << ", " << epsilon_angle_deg << " deg): "
			<< train_accuracy << endl;
	cout << "Test set accuracy (error < " << epsilon_translation << ", " << epsilon_angle_deg << " deg): "
			<< test_accuracy << endl;

// TODO:
// -for each full run, compute some database info
// -for each partial run, compute descriptors (opencv has this
// "BOWImgDescriptorExtractor", but I guess we can't cache the output of
// that, since the bag of words differs for each dataset
// -for each (full run, partial run):
// 	   get best matches
//     count number of consensus elements
//     TODOTODO: figure out how to evaluate match score
// after this is in place, we can vary the following parameters:
// -classifier (SVM? NN? idk)
// -descriptor (SIFT? SURF? lots of stuff in opencv)
// -distance metric
// -score reweighting

	return 0;
}
