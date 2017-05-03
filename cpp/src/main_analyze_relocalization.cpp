#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <map>
#include <algorithm>
#include <memory>

#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml.hpp"

#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;
using namespace std;

using namespace cv;

namespace sdl {

struct Result {

};

struct Frame {
	int index;
	fs::path depthmapPath, imagePath, pointcloudPath;
	Frame(int index) :
			index(index) {
	}

	void setDepthmapPath(fs::path path) {
		depthmapPath = path;
	}
	void setImagePath(fs::path path) {
		imagePath = path;
	}
	void setPointcloudPath(fs::path path) {
		pointcloudPath = path;
	}
};

class Query {
public:
	Query(const fs::path dir) :
			dataDir(dir) {
	}

	void computeFeatures() {
		// TODO a short sequence contains many keyframes. We should pick one from the middle or something.
		// also the next line is going to throw an exception, because it refers to the parent directory, not to a file.
		cout << "about to fail after opening " << dataDir.string() << endl;
		colorImage = imread(dataDir.string());
		featureDetector->detect(colorImage, keypoints);
	}
	void setFeatureDetector(Ptr<FeatureDetector> feature_detector) {
		featureDetector = feature_detector;
	}

	const Mat& getColorImage() const {
		return colorImage;
	}
	const vector<KeyPoint>& getKeypoints() const {
		return keypoints;
	}

private:
	Ptr<FeatureDetector> featureDetector;
	Mat colorImage;
	vector<KeyPoint> keypoints;

	const fs::path dataDir;
};

class Database {
public:
	Database(const fs::path dir) :
			dataDir(dir), keyframes(new map<int, unique_ptr<Frame>>()) {
	}

	vector<Result> lookup(const Query& query, int num_to_return) {

		Mat bow;
		// apparently compute() may modify the keypoints, because it invokes Feature2D.compute()
		// therefore, store the result locally instead of caching them in the query.
		vector<KeyPoint> keypoints;
		bowExtractor->compute(query.getColorImage(), keypoints, bow);

		vector<int> results;
		classifier->predict(bow, results);

		cout << "closest matching frames: ";
		for (int r : results) {
			cout << r << ", ";
		}
		cout << endl;

		// TODO
		return vector<Result>();
	}

	void setFeatureDetector(Ptr<FeatureDetector> feature_detector) {
		featureDetector = feature_detector;
	}

	void setDescriptorExtractor(Ptr<DescriptorExtractor> descriptor_extractor) {
		descriptorExtractor = descriptor_extractor;
	}
	void setBowExtractor(Ptr<BOWImgDescriptorExtractor> bow_extractor) {
		bowExtractor = bow_extractor;
	}

	void train() {

		readDirectoryContents();

		/*vector<ObdImage> images;
		 vector<char> objectPresent;
		 vocData.getClassImages( trainParams.trainObjClass, CV_OBD_TRAIN, images, objectPresent );*/

		TermCriteria terminate_criterion;
		terminate_criterion.epsilon = FLT_EPSILON;
		int vocab_size = 10;
		BOWKMeansTrainer bow_trainer(vocab_size, terminate_criterion);

		cout << "computing descriptors for each keyframe..." << endl;
		// iterate through the images
		map<int, Mat> colorImages;
		for (const auto& element : *keyframes) {
			const auto& keyframe = element.second;
			colorImages[keyframe->index] = imread(keyframe->imagePath.string());

			vector<KeyPoint> imageKeypoints;
			featureDetector->detect(colorImages[keyframe->index],
					imageKeypoints);
			Mat imageDescriptors;
			descriptorExtractor->compute(colorImages[keyframe->index],
					imageKeypoints, imageDescriptors);

			if (!imageDescriptors.empty()) {
				int descriptor_count = imageDescriptors.rows;

				for (int i = 0; i < descriptor_count; i++) {
					bow_trainer.add(imageDescriptors.row(i));
				}
			}
		}
		cout << "computed " << bow_trainer.descriptorsCount() << " descriptors."
				<< endl;

		cout << "Training vocabulary..." << endl;
		vocabulary = bow_trainer.cluster();
		bowExtractor->setVocabulary(vocabulary);

		// Create training data by converting each keyframe to a bag of words
		Mat samples((int) keyframes->size(), vocabulary.rows, CV_32FC1);
		Mat labels((int) keyframes->size(), 1, CV_32SC1);
		int row = 0;
		for (const auto& element : *keyframes) {
			const auto& keyframe = element.second;

			vector<KeyPoint> imageKeypoints;
			Mat bow;
			bowExtractor->compute(colorImages[keyframe->index], imageKeypoints,
					bow);
			bow.copyTo(samples.row(row));
			labels.at<int>(row) = keyframe->index;

			++row;
		}

		classifier = ml::KNearest::create();
		classifier->train(samples, ml::ROW_SAMPLE, labels);

	}
private:
	const fs::path dataDir;

	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<FeatureDetector> featureDetector;

	Mat vocabulary;
	Ptr<BOWImgDescriptorExtractor> bowExtractor;
	Ptr<ml::KNearest> classifier;

	unique_ptr<map<int, unique_ptr<Frame>>> keyframes;

	void readDirectoryContents() {
		for (const auto& file : fs::recursive_directory_iterator(dataDir)) {

			String filename = file.path().filename().string();

			unsigned int delim_idx = filename.find("_");
			if (delim_idx == string::npos) {
				throw runtime_error(
						"unexpected filename while loading database frames! "
						+ file.path().string());
			}
			string stemmed = file.path().filename().stem().string();
			int framenum = stoi(stemmed.substr(delim_idx+1));
			// only insert a new frame if we haven't seen it before
			if (keyframes->find(framenum) == keyframes->end()) {
				keyframes->insert(make_pair(framenum, unique_ptr<Frame>(new Frame(framenum))));
			}

			if (filename.find("cloud") == 0) {
				// pointcloud
				keyframes->at(framenum)->setPointcloudPath(file);
			} else if (filename.find("depth") == 0) {
				// depth map
				keyframes->at(framenum)->setDepthmapPath(file);
			} else if (filename.find("raw") == 0) {
				// image
				keyframes->at(framenum)->setImagePath(file);
			} else {
				throw runtime_error(
						"unexpected filename while loading database frames! "
						+ file.path().string());
			}
		}

	}
};

void initDatabasesAndQueries(const string& directory, vector<Database>& dbs_out,
		vector<vector<Query>>& queries_out) {

	if (!fs::exists(directory) || !fs::is_directory(directory)) {
		cout << "directory " << directory << " does not exist" << endl;
		exit(-1);
	}

	vector<fs::path> trajectory_paths;
	copy(fs::directory_iterator(directory), fs::directory_iterator(),
			back_inserter(trajectory_paths));
	sort(trajectory_paths.begin(), trajectory_paths.end());

	dbs_out.reserve(trajectory_paths.size());
	queries_out.reserve(trajectory_paths.size());
	for (const auto& trajectory_path : trajectory_paths) {

		dbs_out.emplace_back(trajectory_path / "full");
		vector<Query> curQueries;

		fs::directory_iterator dir_end;
		for (auto subdir = fs::directory_iterator(trajectory_path);
				subdir != dir_end; ++subdir) {
			if (!fs::is_directory(subdir->path())) {
				continue;
			}
			if (subdir->path().filename().compare("full") == 0) {
				continue;
			}
			curQueries.emplace_back(subdir->path());
		}

		queries_out.push_back(curQueries);
	}
}
}

int main(int argc, char** argv) {

	if (argc < 2) {
		cout << "usage: " << argv[0] << " <dataset>" << endl;
		cout << "<dataset> is the output directory of main_run_semidense"
				<< endl;
		return -1;
	}

	string datadir(argv[1]);
	vector<sdl::Database> dbs;
	vector<vector<sdl::Query>> queries;
	sdl::initDatabasesAndQueries(datadir, dbs, queries);

	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	Ptr<BOWImgDescriptorExtractor> bow_extractor = makePtr<
			BOWImgDescriptorExtractor>(sift,
			DescriptorMatcher::create("BruteForce"));

	for (auto& db : dbs) {
		db.setDescriptorExtractor(sift);
		db.setFeatureDetector(sift);
		db.setBowExtractor(bow_extractor);
		db.train();
	}
	for (vector<sdl::Query>& queryvec : queries) {
		for (auto& query : queryvec) {
			query.setFeatureDetector(sift);
			query.computeFeatures();
		}
	}

	int num_to_return = 10;
	for (unsigned int i = 0; i < dbs.size(); i++) {
		for (unsigned int j = 0; j < queries.size(); j++) {
			if (i == j) {
				cout << "testing a query on its original sequence!" << endl;
			}

			for (const sdl::Query& q : queries[j]) {
				vector<sdl::Result> results = dbs[i].lookup(q, num_to_return);
				// TODO: analyzer quality of results somehow
			}

		}
	}

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
