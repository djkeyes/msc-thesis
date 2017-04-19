#ifndef SRC_MAPGENERATOR_H_
#define SRC_MAPGENERATOR_H_

#include <string>
#include <memory>
#include <utility>
#include <list>

#include "util/NumType.h"

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
	virtual void savePointCloudAsPly() = 0;
	virtual void saveDepthMaps() = 0;
};

/*
 * Generates a map representation using DSO. This is largely copied from the
 * original DSO command line interface, and accepts similar arguments.
 */
class DsoMapGenerator: public MapGenerator {
public:
	DsoMapGenerator(int argc, char** argv);

	void runVisualOdometry() override;
	void savePointCloudAsPly() override;
	void saveDepthMaps() override;
private:
	void parseArgument(char* arg);

	std::string vignette = "";
	std::string gammaCalib = "";
	std::string source = "";
	std::string calib = "";

	int mode = 0;

	std::shared_ptr<std::list<std::pair<dso::Vec3, float>>>pointcloud;
};

class ArtificialMapGenerator: public MapGenerator {
public:
	ArtificialMapGenerator();

	void runVisualOdometry() override;
	void savePointCloudAsPly() override;
	void saveDepthMaps() override;
};

}

#endif /* SRC_MAPGENERATOR_H_ */
