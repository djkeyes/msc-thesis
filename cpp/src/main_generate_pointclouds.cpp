#include <cstdio>
#include <cstring>
#include <memory>

#include "MapGenerator.h"

using sdl::MapGenerator;
using sdl::DsoMapGenerator;
using sdl::ArtificialMapGenerator;

using std::shared_ptr;
using std::printf;
using std::strcmp;

void usage() {
	// TODO
	printf("Incorrect usage!\n");
}
int main(int argc, char** argv) {

	if (argc < 3) {
		usage();
	}

	bool asPly = false, asPcd = false, asManyPcds = false;
	if (strcmp(argv[2], "ply") == 0) {
		asPly = true;
	} else if (strcmp(argv[2], "pcd") == 0) {
		asPcd = true;
	} else if (strcmp(argv[2], "manypcd") == 0) {
		asManyPcds = true;
	} else {
		usage();
		return 0;
	}

	shared_ptr<MapGenerator> map_gen;
	if (strcmp(argv[1], "dso") == 0) {
		map_gen = shared_ptr<MapGenerator>(
				new DsoMapGenerator(argc - 3, argv + 3));
	} else if (strcmp(argv[1], "simulated") == 0) {
		map_gen = shared_ptr<MapGenerator>(new ArtificialMapGenerator());
	} else {
		usage();
		return 0;
	}

	map_gen->runVisualOdometry();
	if (asPly) {
		map_gen->savePointCloudAsPly();
	} else if (asPcd) {
		map_gen->savePointCloudAsPcd();
	} else if (asManyPcds) {
		map_gen->savePointCloudAsManyPcds();
	}

	return 0;
}
