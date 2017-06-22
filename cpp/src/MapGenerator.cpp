#include "MapGenerator.h"

#include <stdlib.h>
#include <time.h>
#include <vector>
#include <memory>
#include <set>
#include <list>
#include <limits>
#include <utility>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

#include "boost/filesystem.hpp"
#include "util/settings.h"
#include "util/DatasetReader.h"
#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include "IOWrapper/Output3DWrapper.h"

using namespace dso;
using namespace std;
using IOWrap::Output3DWrapper;

namespace sdl {

bool print_debug_info = false;

/*
 * Allows you to read from a TUM monoVO dataset using DSO's ImageFolderReader
 */
class DefaultTumReader: public DsoDatasetReader {
public:
	DefaultTumReader(const string& source, const string& calib, const string& gamma_calib, const string& vignette) :
			reader(source, calib, gamma_calib, vignette) {
		reader.setGlobalCalibration();
		if (setting_photometricCalibration > 0 && reader.getPhotometricGamma() == 0) {
			printf("ERROR: don't have photometric calibation. Need to use commandline options mode=1 ");
			exit(1);
		}
	}

	~DefaultTumReader() = default;

	virtual float* getPhotometricGamma() {
		return reader.getPhotometricGamma();
	}

	virtual int getNumImages() {
		return reader.getNumImages();
	}
	virtual ImageAndExposure* getImage(int id) {
		return reader.getImage(id);
	}

	ImageFolderReader reader;
};

/*
 * Read a dataset specified simply as a list of files, plus camera calibration. This assumes the files have already been
 * somewhat rectified so that the calibration matrix is simply a pinhole camera model.
 */
class SimpleReader: public DsoDatasetReader {
public:
	SimpleReader(const vector<string>& image_paths, cv::Mat K, int width, int height, string tmp_dir) :
			imagePaths(image_paths), dummyPhotometricGamma(255) {
		for (unsigned int i = 0; i < dummyPhotometricGamma.size(); i++) {
			dummyPhotometricGamma[i] = i;
		}

		boost::filesystem::create_directories(tmp_dir);
		createUndistorterFromTmpFile(K, width, height, tmp_dir + "/calib.txt");

		// need to perform global initialization
		cv::Mat K_;
		K.convertTo(K_, CV_32F);
		Eigen::Map<Eigen::Matrix3f> K_as_eigen((float*) K_.data);
		setGlobalCalib(width, height, K_as_eigen);
	}
	~SimpleReader() = default;

	virtual float* getPhotometricGamma() {
		return &dummyPhotometricGamma[0];
	}
	virtual int getNumImages() {
		return imagePaths.size();
	}
	virtual ImageAndExposure* getImage(int id) {
		unique_ptr<MinimalImageB> min_img(IOWrap::readImageBW_8U(imagePaths[id].c_str()));
		ImageAndExposure* ret2 = pinholeUndistorter->undistort<unsigned char>(min_img.get(), 1.0f, 0.0);
		return ret2;
	}

private:

	void createUndistorterFromTmpFile(cv::Mat K, int width, int height, const string& tmp_file) {

		ofstream ofs(tmp_file);
		cv::Mat K_;
		K.convertTo(K_, CV_64F);

		// Pinhole fx fy cx cy 0
		ofs << "Pinhole " << K_.at<double>(0, 0) << " " << K_.at<double>(1, 1) << " " << K_.at<double>(0, 2) << " "
				<< K_.at<double>(1, 2) << " 0" << endl;
		// in_width in_height
		ofs << width << " " << height << endl;
		// "crop" / "full" / "none" / "fx fy cx cy 0"
		ofs << "none" << endl;
		// out_width out_height
		ofs << width << " " << height << endl;
		ofs.close();
		pinholeUndistorter = unique_ptr<Undistort>(new UndistortPinhole(tmp_file.c_str(), false));

		// this produces a lot of chitchat on stdout, but there's not a good way to silence it without
		// completely re-implementing Undistort :(
		pinholeUndistorter->loadPhotometricCalibration("", "", "");
	}
	const vector<string> imagePaths;
	vector<float> dummyPhotometricGamma;
	unique_ptr<Undistort> pinholeUndistorter;
};

struct DsoOutputRecorder: public Output3DWrapper {
	unique_ptr<list<ColoredPoint>> coloredPoints;
	unique_ptr<map<int, pair<SE3, unique_ptr<list<ColoredPoint>>> > > pointsWithViewpointsById;
	unique_ptr<map<int, unique_ptr<MinimalImageF>>> depthImagesById;
	unique_ptr<map<int, unique_ptr<MinimalImageF>>> rgbImagesById;
	unique_ptr<map<int, SE3*>> posesById;
	unique_ptr<map<int, cv::SparseMat>> sceneCoordinateMaps;

	const double my_scaledTH = 1; //1e10;
	const double my_absTH = 1;//1e10;
	const double my_minRelBS = 0.005;// 0;

	DsoOutputRecorder() :
	coloredPoints(new list<ColoredPoint>()),
	pointsWithViewpointsById(new map<int, pair<SE3, unique_ptr<list<ColoredPoint>>> >()),
	depthImagesById(new map<int, unique_ptr<MinimalImageF>>()),
	rgbImagesById(new map<int, unique_ptr<MinimalImageF>>()),
	posesById(new map<int, SE3*>()),
	sceneCoordinateMaps(new map<int, cv::SparseMat>()) {
	}

	void publishGraph(
			const map<uint64_t, Eigen::Vector2i, less<uint64_t>,
			Eigen::aligned_allocator<pair<uint64_t, Eigen::Vector2i> > > &connectivity)
	override {
		// TODO(daniel): we could check the graph connectivity here to get covisibility
		// of keypoints. This is a little sketchy though, since DSO is a direct method,
		// so we aren't guaranteed that keypoint k in frame i will also project to a
		// keypoint in frame j. On the other hand, if it doesn't project to a keypoint,
		// we could just use that as an excuse to add a new keypoint to frame j.
	}

	void printPoint(const PointHessian* const & point) {
		if(print_debug_info) {
			printf("\tcolor=[%f, %f, %f, %f, %f, %f, %f, %f], "
					"uv=[%f, %f], idepth=%f, idepth_backup=%f, "
					"idepth_hessian=%f, idepth_scaled=%f, "
					"idepth_zero=%f, idepth_zero_scaled=%f, "
					"idx=%d, instanceCounter=%d, status=%d\n", point->color[0],
					point->color[1], point->color[2], point->color[3],
					point->color[4], point->color[5], point->color[6],
					point->color[7], point->u, point->v, point->idepth,
					point->idepth_backup, point->idepth_hessian,
					point->idepth_scaled, point->idepth_zero,
					point->idepth_zero_scaled, point->idx, point->instanceCounter,
					point->status);
		}}

	void addPoints(const vector<PointHessian*>& points, CalibHessian* calib,
			const SE3& camToWorld, int frame_id) {

		double fxi = calib->fxli();
		double fyi = calib->fyli();
		double cxi = calib->cxli();
		double cyi = calib->cyli();

		unique_ptr<list<ColoredPoint>> current_points(new list<ColoredPoint>());
		int dims[] = {hG[0], wG[0]};
		cv::SparseMat sceneCoordMap;
		sceneCoordMap.create(2, dims,  CV_32FC3);

		for (PointHessian* point : points) {
			double z = 1.0 / point->idepth;

			// check some bounds to avoid crazy outliers
			if (z < 0) {
				continue;
			}
			float z4 = z * z * z * z;
			float var = (1.0f / (point->idepth_hessian + 0.01));
			if (var * z4 > my_scaledTH) {
				if(print_debug_info) {
					printf("skipping because large var (%f) * z^4 (%f) = %f > %f\n",
							var, z4, var * z4, my_scaledTH);
				}
				continue;
			}
			if (var > my_absTH) {
				if(print_debug_info) {
					printf("skipping because large var (%f) > %f\n", var, my_absTH);}
				continue;
			}
			if (point->maxRelBaseline < my_minRelBS) {
				if(print_debug_info) {
					printf("skipping because large maxRelBaseline (%f) < %f\n",
							point->maxRelBaseline, my_minRelBS);
				}
				continue;
			}

			for (int patternIdx = 0; patternIdx < patternNum; patternIdx++) {
				int dx = patternP[patternIdx][0];
				int dy = patternP[patternIdx][1];

				if(dx != 0 || dy != 0) {
					continue;
				}

				double x = ((point->u+dx)*fxi + cxi) * z;
				double y = ((point->v+dy)*fyi + cyi) * z;

				float color = point->color[patternIdx] / 255.0;

				Vec3 point3d(x, y, z);

				coloredPoints->push_back(make_pair(camToWorld*point3d, color));
				current_points->push_back(make_pair(point3d, color));

				sceneCoordMap.ref<cv::Vec3f>(static_cast<int>(point->v), static_cast<int>(point->u)) = cv::Vec3f(x,y,z);
			}
		}
		pointsWithViewpointsById->insert(make_pair(frame_id, make_pair(camToWorld, move(current_points))));
		sceneCoordinateMaps->insert(make_pair(frame_id, sceneCoordMap));
	}

	void publishKeyframes(vector<FrameHessian*> &frames, bool final,
			CalibHessian* HCalib) override {

		if (final) {
			FrameHessian* fh = frames[0];
			// fh contains points in pointHessians, pointHessiansMarginalized, pointHessiansOut, and ImmaturePoints
			// but since it's final, this indicates there are no points in pointHessians (which has "active points").
			// for now, just project pointHessiansMarginalized and pointHessiansOut (ImmaturePoints has large
			// uncertainties, but we could experiment with those, too)
			addPoints(fh->pointHessiansMarginalized, HCalib, fh->PRE_camToWorld, fh->shell->incoming_id);
//			addPoints(fh->pointHessiansOut, HCalib, fh->PRE_camToWorld, fh->shell->incoming_id);

			// also extract and save the image
			int width = wG[0];
			int height = hG[0];
			unique_ptr<MinimalImageF> image(new MinimalImageF(width, height));
			for(int i=0; i < width*height; i++) {
				// I think the first component of this just contains the raw images.
				// dI[i][j] contains the ith pyrimidal level and the gradient in
				// the jth direction, for i, j >= 1
				image->data[i] = fh->dI[i][0];
			}
			rgbImagesById->insert(make_pair(fh->shell->incoming_id, move(image)));
		}
	}

	void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override {
		posesById->insert(make_pair(frame->incoming_id, &frame->camToWorld));
	}

	void pushLiveFrame(FrameHessian* image) override {
//		printf("pushLiveFrame() called!\n");
	}

	void pushDepthImage(MinimalImageB3* image) override {
	}
	bool needPushDepthImage() override {
		return false;
	}

	void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF) override {
		int id = KF->shell->incoming_id;
		// make a copy of the image, otherwise might be erased
		depthImagesById->insert(make_pair(id, unique_ptr<MinimalImageF>(image->getClone())));
	}

	void join() override {
//		printf("join() called!\n");
	}

	void reset() override {
//		printf("reset() called!\n");
	}

};

// TODO: DsoMapGenerator has a lot of global state, since it's sort of coupled
// with the variables in util/settings.h. Can we wrap that somehow?

DsoMapGenerator::DsoMapGenerator(int argc, char** argv) {
	setting_desiredImmatureDensity = 1500;
	setting_desiredPointDensity = 2000;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations = 6;
	setting_minOptIterations = 1;

	if (print_debug_info) {
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- %f active points\n"
				"- %d-%d active frames\n"
				"- %d-%d LM iteration each KF\n"
				"- original image resolution\n"
				"See MapGenerator.cpp to expose more settings\n", "not",
				setting_desiredPointDensity, setting_minFrames,
				setting_maxFrames, setting_minOptIterations,
				setting_maxOptIterations);
	}

	setting_logStuff = false;

	disableAllDisplay = true;

	string source, calib, gamma_calib, vignette;
	for (int i = 0; i < argc; i++) {
		parseArgument(argv[i], source, calib, gamma_calib, vignette);
	}

	datasetReader = unique_ptr<DsoDatasetReader>(
			new DefaultTumReader(source, calib, gamma_calib, vignette));
}
DsoMapGenerator::DsoMapGenerator(const string& input_path) {
	setting_desiredImmatureDensity = 3000;
	setting_desiredPointDensity = 4000;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations = 6;
	setting_minOptIterations = 1;

	setting_logStuff = false;

	disableAllDisplay = true;

	setting_debugout_runquiet = true;

	// to handle datasets other than tum monoVO, we'll need to change these
	// paths, and change the mode and photometric calibration weights
	string source = input_path + "/images/";
	string calib = input_path + "/camera.txt";
	string gamma_calib = input_path + "/pcalib.txt";
	string vignette = input_path + "/vignette.png";

	mode = 0;

	datasetReader = unique_ptr<DsoDatasetReader>(
			new DefaultTumReader(source, calib, gamma_calib, vignette));
}

/*
 * Create an instance using just a camera calibration. Geometric calibration
 * is disabled, gamma is set to a default value, and all frames are treated
 * as keyframes.
 */
DsoMapGenerator::DsoMapGenerator(cv::Mat camera_calib, int width, int height, const vector<string>& image_paths, const string& cache_path) {
	setting_desiredImmatureDensity = 3000;
	setting_desiredPointDensity = 4000;
	// setting this higher than 1 forces more frames to be keyframes. Still, many frames
	// are skipped for a variety of reasons. The most common is probably no camera
	// movement / ill-conditioned depth estimation.
	setting_kfGlobalWeight = 2;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations = 6;
	setting_minOptIterations = 1;
	setting_realTimeMaxKF = true;

	setting_logStuff = false;

	disableAllDisplay = true;

	setting_debugout_runquiet = true;

	// to handle datasets other than tum monoVO, we'll need to change these
	// paths, and change the mode and photometric calibration weights
	datasetReader = unique_ptr<DsoDatasetReader>(new SimpleReader(image_paths, camera_calib, width, height, cache_path));

	setting_photometricCalibration = 0;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;

	mode = 1;
}


void DsoMapGenerator::runVisualOdometry() {
	vector<int> idsToPlay;
	for (int i = 0; i < datasetReader->getNumImages(); ++i) {
		idsToPlay.push_back(i);
	}
	runVisualOdometry(idsToPlay);
}
void DsoMapGenerator::runVisualOdometry(const vector<int>& ids_to_play) {

	unique_ptr<FullSystem> fullSystem(new FullSystem());
	fullSystem->setGammaFunction(datasetReader->getPhotometricGamma());
	unique_ptr<DsoOutputRecorder> dso_recorder(new DsoOutputRecorder());
	fullSystem->outputWrapper.push_back(dso_recorder.get());

	clock_t started = clock();

	for (int i = 0; i < static_cast<int>(ids_to_play.size()); i++) {
		if (!fullSystem->initialized)	// if not initialized: reset start time.
		{
			started = clock();
		}

		int id = ids_to_play[i];

		ImageAndExposure* img = datasetReader->getImage(id);

		fullSystem->addActiveFrame(img, id);

		delete img;

		if (fullSystem->initFailed || setting_fullResetRequested) {
			if (i < 250 || setting_fullResetRequested) {
				if (print_debug_info) {
					printf("RESETTING!\n");
				}

				vector<IOWrap::Output3DWrapper*> wraps =
						fullSystem->outputWrapper;

				fullSystem = unique_ptr<FullSystem>(new FullSystem());
				fullSystem->setGammaFunction(datasetReader->getPhotometricGamma());

				for (IOWrap::Output3DWrapper* ow : wraps)
					ow->reset();
				fullSystem->outputWrapper = wraps;

				setting_fullResetRequested = false;
			}
		}

		if (fullSystem->isLost) {
			if (print_debug_info) {
				printf("LOST!!\n");
			}
			break;
		}

	}
	fullSystem->blockUntilMappingIsFinished();
	clock_t ended = clock();

	fullSystem->printResult("result.txt");

	int numFramesProcessed = ids_to_play.size();
	double MilliSecondsTakenSingle = 1000.0f * (ended - started)
			/ (float) (CLOCKS_PER_SEC);
	if (print_debug_info) {
		printf("\n======================"
				"\n%d Frames"
				"\n%.2fms total"
				"\n%.2fms per frame"
				"\n======================\n\n", numFramesProcessed,
				MilliSecondsTakenSingle,
				MilliSecondsTakenSingle / numFramesProcessed);
	}

	pointcloud = move(dso_recorder->coloredPoints);
	pointcloudsWithViewpoints = move(dso_recorder->pointsWithViewpointsById);
	depthImages = move(dso_recorder->depthImagesById);
	rgbImages = move(dso_recorder->rgbImagesById);
	poses = move(dso_recorder->posesById);
	sceneCoordinateMaps = move(dso_recorder->sceneCoordinateMaps);

	if (print_debug_info) {
		printf("recorded %lu points!\n", pointcloud->size());
	}

	for (IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper) {
		ow->join();
	}
}
void DsoMapGenerator::savePointCloudAsPly(const string& filename) {
	string path(filename.substr(0, filename.rfind("/") + 1));
	// create directory if it doesn't exist
	boost::filesystem::path dir(path.c_str());
	boost::filesystem::create_directories(dir);

	ofstream out;
	out.open(filename, ios::out | ios::trunc | ios::binary);
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "element vertex " << pointcloud->size() << "\n";

	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property float intensity\n";
	out << "end_header\n";
	for (const ColoredPoint& p : *pointcloud) {
		// TODO: we lose precision here, but meshlab won't work if we use a double. why?
		float x = (float) p.first.x();
		float y = (float) p.first.y();
		float z = (float) p.first.z();

		out.write((const char*) &x, sizeof(float));
		out.write((const char*) &y, sizeof(float));
		out.write((const char*) &z, sizeof(float));
		out.write((const char*) &p.second, sizeof(float));
	}

	out.close();

}

pcl::PointCloud<pcl::PointXYZI> convertListToCloud(
		const list<ColoredPoint>& cloudAsList) {
	pcl::PointCloud<pcl::PointXYZI> cloud;
	cloud.width = cloudAsList.size();
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width * cloud.height);

	auto iter = cloudAsList.begin();
	for (size_t i = 0; i < cloud.points.size(); ++i, ++iter) {
		cloud.points[i].x = iter->first.x();
		cloud.points[i].y = iter->first.y();
		cloud.points[i].z = iter->first.z();
		cloud.points[i].intensity = iter->second;
	}
	return cloud;
}
void DsoMapGenerator::savePointCloudAsPcd(const string& filename) {
	string path(filename.substr(0, filename.rfind("/") + 1));
	// create directory if it doesn't exist
	boost::filesystem::path dir(path.c_str());
	boost::filesystem::create_directories(dir);

	pcl::PointCloud<pcl::PointXYZI> cloud = convertListToCloud(*pointcloud);
	pcl::io::savePCDFileBinary(filename, cloud);
}
void DsoMapGenerator::savePointCloudAsManyPcds(const string& filepath) {

	string path(filepath.substr(0, filepath.rfind("/") + 1));
	// create directory if it doesn't exist
	boost::filesystem::path dir(path.c_str());
	boost::filesystem::create_directories(dir);

	for (auto const& element : *pointcloudsWithViewpoints) {
		// if no points, skip
		// (it would be nice to just save an empty pointcloud, but PCL throws an exception)
		if (element.second.second->empty()) {
			continue;
		}

		int id = element.first;

		const SE3& viewpoint = element.second.first;

		pcl::PointCloud<pcl::PointXYZI> cloud = convertListToCloud(
				*element.second.second);
		cloud.sensor_origin_.head<3>() = viewpoint.translation().cast<float>();
		cloud.sensor_origin_.w() = 1;

		cloud.sensor_orientation_ =
				viewpoint.so3().unit_quaternion().cast<float>();

		stringstream filename_stream;
		filename_stream << filepath << "_" << id << ".pcd";
		pcl::io::savePCDFileBinary(filename_stream.str(), cloud);

	}

}
void DsoMapGenerator::saveDepthMaps(const string& filepath) {
	string path(filepath.substr(0, filepath.rfind("/") + 1));
	// create directory if it doesn't exist
	boost::filesystem::path dir(path.c_str());
	boost::filesystem::create_directories(dir);

	for (auto const& element : *depthImages) {
		int id = element.first;
		MinimalImageF* depthMap = element.second.get();

		stringstream filename_stream;
		filename_stream << filepath << "_" << id << ".png";
		IOWrap::writeImage(filename_stream.str(), depthMap);
	}

}
void DsoMapGenerator::savePosesInWorldFrame(const string& gt_filename,
		const string& output_filename) const {

	ifstream ground_truth;
	ground_truth.open(gt_filename);
	int firstTrackedFrame = 0;
	SE3 trajectory_to_world;

	for (;; firstTrackedFrame++) {

		double time;
		double x, y, z;
		double qx, qy, qz, qw;

		// expecting format 'time x y z qx qy qz qw', where x thru qw can be NaN
		string line, tmp;
		getline(ground_truth, line);
		stringstream ss(line);
		ss >> time;
		ss >> tmp;

		// apparently operator>> can't parse the string "NaN" correctly
		// skip if the current line has NaN, or we don't have a pose estimate.
		if (tmp.find("NaN") != tmp.npos
				|| poses->find(firstTrackedFrame) == poses->end()) {
			continue;
		}
		ss >> y >> z >> qx >> qy >> qz >> qw;
		ss = stringstream(tmp);
		ss >> x;

		Eigen::Quaternion<SE3::Scalar> orientation(qw, qx, qy, qz);
		SE3::Point translation(x, y, z);

		SE3 world_to_camera(orientation, translation);

		trajectory_to_world = world_to_camera
				* poses->at(firstTrackedFrame)->inverse();

		break;
	}

	string path(output_filename.substr(0, output_filename.rfind("/") + 1));
	// create directory if it doesn't exist
	boost::filesystem::path dir(path.c_str());
	boost::filesystem::create_directories(dir);

	ofstream out;
	out.open(output_filename, ios::out | ios::trunc);
	for (const auto& element : *poses) {
		int id = element.first;
		SE3 pose = trajectory_to_world * (*element.second);

		out << id << " " << pose.translation().x() << " "
				<< pose.translation().y() << " " << pose.translation().z()
				<< " " << pose.unit_quaternion().x() << " "
				<< pose.unit_quaternion().y() << " "
				<< pose.unit_quaternion().z() << " "
				<< pose.unit_quaternion().w() << endl;
	}
	out.close();
}
void DsoMapGenerator::saveRawImages(const string& filepath) const {
	string path(filepath.substr(0, filepath.rfind("/") + 1));
	// create directory if it doesn't exist
	boost::filesystem::path dir(path.c_str());
	boost::filesystem::create_directories(dir);

	for (auto const& element : *rgbImages) {
		int id = element.first;
		MinimalImageF* image = element.second.get();

		stringstream filename_stream;
		filename_stream << filepath << "_" << id << ".png";
		IOWrap::writeImage(filename_stream.str(), image);
	}
}

void DsoMapGenerator::parseArgument(char* arg, string& source, string& calib, string& gamma_calib, string& vignette) {
	int option;
	char buf[1000];

	if (1 == sscanf(arg, "quiet=%d", &option)) {
		if (option == 1) {
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}

	if (1 == sscanf(arg, "rec=%d", &option)) {
		if (option == 0) {
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}

	if (1 == sscanf(arg, "nolog=%d", &option)) {
		if (option == 1) {
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if (1 == sscanf(arg, "nomt=%d", &option)) {
		if (option == 1) {
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}

	if (1 == sscanf(arg, "files=%s", buf)) {
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}

	if (1 == sscanf(arg, "calib=%s", buf)) {
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}

	if (1 == sscanf(arg, "vignette=%s", buf)) {
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if (1 == sscanf(arg, "gamma=%s", buf)) {
		gamma_calib = buf;
		printf("loading gammaCalib from %s!\n", gamma_calib.c_str());
		return;
	}

	if (1 == sscanf(arg, "save=%d", &option)) {
		if (option == 1) {
			debugSaveImages = true;
			if (42 == system("rm -rf images_out"))
				printf(
						"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if (42 == system("mkdir images_out"))
				printf(
						"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if (42 == system("rm -rf images_out"))
				printf(
						"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if (42 == system("mkdir images_out"))
				printf(
						"system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if (1 == sscanf(arg, "mode=%d", &option)) {

		mode = option;
		if (option == 0) {
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if (option == 1) {
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}

int DsoMapGenerator::getNumImages() {
	return datasetReader->getNumImages();
}

map<int, cv::SparseMat> DsoMapGenerator::getSceneCoordinateMaps() {
	// returns a copy
	return *sceneCoordinateMaps;
}

}
