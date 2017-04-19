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
}
int main(int argc, char** argv) {

	if (argc < 2) {
		usage();
	}

	shared_ptr<MapGenerator> map_gen;
	if (strcmp(argv[1], "dso") == 0) {
		map_gen = shared_ptr<MapGenerator>(new DsoMapGenerator(argc, argv));
	} else if (strcmp(argv[1], "simulated") == 0) {
		map_gen = shared_ptr<MapGenerator>(new ArtificialMapGenerator());
	} else {
		usage();
		return 0;
	}

	map_gen->runVisualOdometry();
	map_gen->savePointCloudAsPly();

	return 0;
}
