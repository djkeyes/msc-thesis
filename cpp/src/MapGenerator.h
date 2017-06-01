#ifndef SRC_MAPGENERATOR_H_
#define SRC_MAPGENERATOR_H_

#include <string>
#include <memory>
#include <utility>
#include <list>

#include "opencv2/core/mat.hpp"

#include "util/NumType.h"
#include "util/DatasetReader.h"
#include "util/MinimalImage.h"

namespace sdl {

/*
 * Abstract class to generate and save a representation of a map, whether as a
 * point cloud, collections of poses + depth maps, or otherwise.
 */
class MapGenerator {

public:
	virtual ~MapGenerator() {
	}

	virtual void runVisualOdometry() = 0;
	virtual void savePointCloudAsPly(const std::string& filename) = 0;
	virtual void savePointCloudAsPcd(const std::string& filename) = 0;
	virtual void savePointCloudAsManyPcds(const std::string& filepath) = 0;
	virtual void saveDepthMaps(const std::string& filepath) = 0;
};

typedef std::pair<dso::Vec3, float> ColoredPoint;

/*
 * Generates a map representation using DSO. This is largely copied from the
 * original DSO command line interface, and accepts similar arguments.
 */
class DsoMapGenerator: public MapGenerator {
public:

	DsoMapGenerator(int argc, char** argv);
	DsoMapGenerator(const std::string& input_path);
	DsoMapGenerator(cv::Mat camera_calib, const std::string& image_path, const std::string& cache_path);

	void initVisualOdometry();
	void runVisualOdometry(const std::vector<int>& indices_to_play);
	void runVisualOdometry() override;
	inline bool hasValidPoints() const {
		return !pointcloud->empty();
	}

	void savePointCloudAsPly(const std::string& filename) override;
	void savePointCloudAsPcd(const std::string& filename) override;
	void savePointCloudAsManyPcds(const std::string& filepath) override;
	void saveDepthMaps(const std::string& filepath) override;

	void saveRawImages(const std::string& filepath) const;
	void savePosesInWorldFrame(const std::string& gt_filename, const std::string& output_filename) const;


	inline ImageFolderReader& getReader() {
		return *reader;
	}

private:
	void parseArgument(char* arg);

	std::string vignette = "";
	std::string gammaCalib = "";
	std::string source = "";
	std::string calib = "";

	int mode = 0;

	std::unique_ptr<std::list<ColoredPoint>> pointcloud;
	std::unique_ptr<
			std::map<int,
					std::pair<dso::SE3,
							std::unique_ptr<std::list<ColoredPoint>>> > >pointcloudsWithViewpoints;
	std::unique_ptr<std::map<int, std::unique_ptr<dso::MinimalImageF>>> depthImages;
	std::unique_ptr<std::map<int, std::unique_ptr<dso::MinimalImageF>>> rgbImages;
	std::unique_ptr<std::map<int, dso::SE3*>> poses;

	std::unique_ptr<ImageFolderReader> reader;
};

class ArtificialMapGenerator: public MapGenerator {
public:
	ArtificialMapGenerator();

	void runVisualOdometry() override;
	void savePointCloudAsPly(const std::string& filename) override;
	void savePointCloudAsPcd(const std::string& filename) override;
	void savePointCloudAsManyPcds(const std::string& filepath) override;
	void saveDepthMaps(const std::string& filepath) override;
};

}

#endif /* SRC_MAPGENERATOR_H_ */
