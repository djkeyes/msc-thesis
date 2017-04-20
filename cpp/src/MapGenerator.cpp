#include "MapGenerator.h"

#include <stdlib.h>
#include <time.h>
#include <vector>
#include <memory>
#include <set>
#include <list>
#include <utility>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

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

struct DsoOutputRecorder: public Output3DWrapper {
	shared_ptr<list<ColoredPoint>> coloredPoints;
	shared_ptr<map<int, pair<SE3, shared_ptr<list<ColoredPoint>>> > > pointsWithViewpointsById;

	const double my_scaledTH = 1; //1e10;
	const double my_absTH = 1;//1e10;
	const double my_minRelBS = 0.005;// 0;

	DsoOutputRecorder() :
	coloredPoints(new list<ColoredPoint>()), pointsWithViewpointsById(new map<int, pair<SE3, shared_ptr<list<ColoredPoint>>> >()) {
	}

	void publishGraph(
			const map<uint64_t, Eigen::Vector2i, less<uint64_t>,
			Eigen::aligned_allocator<pair<uint64_t, Eigen::Vector2i> > > &connectivity)
	override {
		printf("publishGraph() called!\n");
	}

	void printPoint(const PointHessian* const & point) {
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
	}

	void addPoints(const vector<PointHessian*>& points, CalibHessian* calib,
			const SE3& camToWorld) {

		double fxi = calib->fxli();
		double fyi = calib->fyli();
		double cxi = calib->cxli();
		double cyi = calib->cyli();

		shared_ptr<list<ColoredPoint>> current_points(new list<ColoredPoint>());
		for (PointHessian* point : points) {
			double z = 1.0 / point->idepth;

			// check some bounds to avoid crazy outliers
			if (z < 0) {
				continue;
			}
			float z4 = z * z * z * z;
			float var = (1.0f / (point->idepth_hessian + 0.01));
			if (var * z4 > my_scaledTH) {
				printf("skipping because large var (%f) * z^4 (%f) = %f > %f\n",
						var, z4, var * z4, my_scaledTH);
				continue;
			}
			if (var > my_absTH) {
				printf("skipping because large var (%f) > %f\n", var, my_absTH);
				continue;
			}
			if (point->maxRelBaseline < my_minRelBS) {
				printf("skipping because large maxRelBaseline (%f) < %f\n",
						point->maxRelBaseline, my_minRelBS);
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

				Vec3 point(x, y, z);

				coloredPoints->push_back(make_pair(camToWorld*point, color));
				current_points->push_back(make_pair(point, color));
			}
		}
		int frame_id = pointsWithViewpointsById->size();
		pointsWithViewpointsById->insert(make_pair(frame_id, make_pair(camToWorld, current_points)));
	}

	void publishKeyframes(vector<FrameHessian*> &frames, bool final,
			CalibHessian* HCalib) override {

		if (final) {
			FrameHessian* fh = frames[0];
			// fh contains points in pointHessians, pointHessiansMarginalized, pointHessiansOut, and ImmaturePoints
			// but since it's final, this indicates there are no points in pointHessians (which has "active points").
			// for now, just project pointHessiansMarginalized and pointHessiansOut (ImmaturePoints has large
			// uncertainties, but we could experiment with those, too)
			addPoints(fh->pointHessiansMarginalized, HCalib, fh->PRE_camToWorld);
//			addPoints(fh->pointHessiansOut, HCalib, fh->PRE_camToWorld);
		}
	}

	void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override {
//		printf("publishCamPose() called!\n");
	}

	void pushLiveFrame(FrameHessian* image) override {
//		printf("pushLiveFrame() called!\n");
	}

	void pushDepthImage(MinimalImageB3* image) override {
//		printf("pushDepthImage() called!\n");
	}
	bool needPushDepthImage() override {
//		printf("needPushDepthImage() called! (returned false)\n");
		return false;
	}

	void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF) {
//		printf("pushDepthImageFloat() called!\n");
	}

	void join() override {
		printf("join() called!\n");
	}

	void reset() override {
		printf("reset() called!\n");
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

	printf("DEFAULT settings:\n"
			"- %s real-time enforcing\n"
			"- %f active points\n"
			"- %d-%d active frames\n"
			"- %d-%d LM iteration each KF\n"
			"- original image resolution\n"
			"See MapGenerator.cpp to expose more settings\n", "not",
			setting_desiredPointDensity, setting_minFrames, setting_maxFrames,
			setting_minOptIterations, setting_maxOptIterations);

	setting_logStuff = false;

	disableAllDisplay = true;

	for (int i = 0; i < argc; i++) {
		parseArgument(argv[i]);
	}
}
void DsoMapGenerator::runVisualOdometry() {

	shared_ptr<ImageFolderReader> reader(
			new ImageFolderReader(source, calib, gammaCalib, vignette));
	reader->setGlobalCalibration();

	if (setting_photometricCalibration > 0
			&& reader->getPhotometricGamma() == 0) {
		printf(
				"ERROR: dont't have photometric calibation. Need to use commandline options mode=1 ");
		exit(1);
	}

	shared_ptr<FullSystem> fullSystem(new FullSystem());
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	unique_ptr<DsoOutputRecorder> dso_recorder(new DsoOutputRecorder());
	fullSystem->outputWrapper.push_back(dso_recorder.get());

	vector<int> idsToPlay;
	for (int i = 0; i < reader->getNumImages(); ++i) {
		idsToPlay.push_back(i);
	}

	clock_t started = clock();

	for (int i = 0; i < (int) idsToPlay.size(); i++) {
		if (!fullSystem->initialized)	// if not initialized: reset start time.
		{
			started = clock();
		}

		int id = idsToPlay[i];

		ImageAndExposure* img = reader->getImage(id);

		fullSystem->addActiveFrame(img, id);

		delete img;

		if (fullSystem->initFailed || setting_fullResetRequested) {
			if (i < 250 || setting_fullResetRequested) {
				printf("RESETTING!\n");

				vector<IOWrap::Output3DWrapper*> wraps =
						fullSystem->outputWrapper;

				fullSystem = shared_ptr<FullSystem>(new FullSystem());
				fullSystem->setGammaFunction(reader->getPhotometricGamma());

				for (IOWrap::Output3DWrapper* ow : wraps)
					ow->reset();
				fullSystem->outputWrapper = wraps;

				setting_fullResetRequested = false;
			}
		}

		if (fullSystem->isLost) {
			printf("LOST!!\n");
			break;
		}

	}
	fullSystem->blockUntilMappingIsFinished();
	clock_t ended = clock();

	fullSystem->printResult("result.txt");

	int numFramesProcessed = idsToPlay.size();
	double numSecondsProcessed = fabs(
			reader->getTimestamp(idsToPlay[0])
					- reader->getTimestamp(idsToPlay.back()));
	double MilliSecondsTakenSingle = 1000.0f * (ended - started)
			/ (float) (CLOCKS_PER_SEC);
	printf("\n======================"
			"\n%d Frames (recorded at %.1f fps)"
			"\n%.2fms total"
			"\n%.2fms per frame"
			"\n======================\n\n", numFramesProcessed,
			numFramesProcessed / numSecondsProcessed, MilliSecondsTakenSingle,
			MilliSecondsTakenSingle / numFramesProcessed);

	pointcloud = shared_ptr<list<ColoredPoint>>(dso_recorder->coloredPoints);
	pointcloudsWithViewpoints = dso_recorder->pointsWithViewpointsById;
	printf("recorded %lu points!\n", pointcloud->size());

	for (IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper) {
		ow->join();
	}

}
void DsoMapGenerator::savePointCloudAsPly(const std::string& filename) {
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
		shared_ptr<list<ColoredPoint> const> cloudAsList) {
	pcl::PointCloud<pcl::PointXYZI> cloud;
	cloud.width = cloudAsList->size();
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width * cloud.height);

	auto iter = cloudAsList->begin();
	for (size_t i = 0; i < cloud.points.size(); ++i, ++iter) {
		cloud.points[i].x = iter->first.x();
		cloud.points[i].y = iter->first.y();
		cloud.points[i].z = iter->first.z();
		cloud.points[i].intensity = iter->second;
	}
	return cloud;
}
void DsoMapGenerator::savePointCloudAsPcd(const std::string& filename) {
	pcl::PointCloud<pcl::PointXYZI> cloud = convertListToCloud(pointcloud);
	pcl::io::savePCDFileBinary(filename, cloud);
}
void DsoMapGenerator::savePointCloudAsManyPcds(const std::string& filepath) {

	for (auto const element : *pointcloudsWithViewpoints) {
		int id = element.first;

		const SE3& viewpoint = element.second.first;

		pcl::PointCloud<pcl::PointXYZI> cloud = convertListToCloud(
				element.second.second);
		cloud.sensor_origin_.head<3>() = viewpoint.translation().cast<float>();
		cloud.sensor_origin_.w() = 1;

		cloud.sensor_orientation_ =
				viewpoint.so3().unit_quaternion().cast<float>();

		stringstream filename_stream;
		filename_stream << filepath << "_" << id << ".pcd";
		pcl::io::savePCDFileBinary(filename_stream.str(), cloud);

	}

}
void DsoMapGenerator::saveDepthMaps(const std::string& filepath) {
}

void DsoMapGenerator::parseArgument(char* arg) {
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
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
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

// TODO: generate artificial data for testing
ArtificialMapGenerator::ArtificialMapGenerator() {
}
void ArtificialMapGenerator::runVisualOdometry() {
}
void ArtificialMapGenerator::savePointCloudAsPly(const std::string& filename) {
}
void ArtificialMapGenerator::savePointCloudAsPcd(const std::string& filename) {
}
void ArtificialMapGenerator::savePointCloudAsManyPcds(
		const std::string& filepath) {
}
void ArtificialMapGenerator::saveDepthMaps(const std::string& filepath) {
}

}
