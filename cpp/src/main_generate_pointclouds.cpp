#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "MapGenerator.h"

using sdl::MapGenerator;
using sdl::DsoMapGenerator;

using std::shared_ptr;
using std::printf;
using std::strcmp;
using std::string;

void usage() {
  // TODO
  printf("Incorrect usage!\n");
}
int main(int argc, char** argv) {
  int num_required_args = 4;
  if (argc < num_required_args) {
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
  string filename(argv[3]);

  shared_ptr<MapGenerator> map_gen;
  if (strcmp(argv[1], "dso") == 0) {
    map_gen = shared_ptr<MapGenerator>(new DsoMapGenerator(
        argc - num_required_args, argv + num_required_args));
  } else {
    usage();
    return 0;
  }

  map_gen->runVisualOdometry();
  if (asPly) {
    map_gen->savePointCloudAsPly(filename);
  } else if (asPcd) {
    map_gen->savePointCloudAsPcd(filename);
  } else if (asManyPcds) {
    map_gen->savePointCloudAsManyPcds(filename);
  }

  return 0;
}
