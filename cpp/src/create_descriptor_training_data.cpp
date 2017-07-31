#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "boost/program_options.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Datasets.h"
#include "Relocalization.h"
#include "MapGenerator.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

void usage(char** argv, const po::options_description commandline_args) {
  cout << "Usage: " << argv[0] << " [options]" << endl;
  cout << commandline_args << "\n";
}

sdl::SceneParser* parseArguments(int argc, char** argv) {
  // Arguments, can be specified on commandline or in a file settings.config
  po::options_description commandline_exclusive(
      "Allowed options from terminal");
  commandline_exclusive.add_options()("help", "Print this help message.")(
      "config", po::value<string>(),
      "Path to a config file, which can specify any other argument.");

  po::options_description general_args(
      "Allowed options from terminal or config file");
  general_args.add_options()
      ("scene", po::value<string>(), "Type of scene. Currently the only allowed type is 7scenes.")
      ("datadir", po::value<string>()->default_value(""), "Directory of the scene dataset. For datasets composed"
          " of several scenes, this should be the appropriate subdirectory.")
      ("cache", po::value<string>()->default_value(""), "Directory to cache intermediate results, ie"
          " descriptors or visual vocabulary, between runs.");

  po::options_description commandline_args;
  commandline_args.add(commandline_exclusive).add(general_args);

  // check for config file
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(commandline_exclusive)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    // print help and exit
    usage(argv, commandline_args);
    exit(0);
  }

  if (vm.count("config")) {
    ifstream ifs(vm["config"].as<string>());
    if (!ifs) {
      cout << "could not open config file " << vm["config"].as<string>()
           << endl;
      exit(1);
    }
    vm.clear();
    // since config is added last, commandline args have precedence over
    // args in the config file.
    po::store(
        po::command_line_parser(argc, argv).options(commandline_args).run(),
        vm);
    po::store(po::parse_config_file(ifs, general_args), vm);
  } else {
    vm.clear();
    po::store(
        po::command_line_parser(argc, argv).options(commandline_args).run(),
        vm);
  }
  po::notify(vm);

  fs::path cache_dir;
  if (vm.count("cache")) {
    cache_dir = fs::path(vm["cache"].as<string>());
  }

  if (vm.count("scene")) {
    string scene_type(vm["scene"].as<string>());
    // currently only one supported scene
    if (scene_type.find("7scenes") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      sdl::SevenScenesParser* parser = new sdl::SevenScenesParser(directory);
      if (!cache_dir.empty()) {
        parser->setCache(cache_dir);
      }
      return parser;
    } else if (scene_type.find("tum") == 0) {
      fs::path directory(vm["datadir"].as<string>());
      sdl::TumParser* parser = new sdl::TumParser(directory);
      if (!cache_dir.empty()) {
        parser->setCache(cache_dir);
      }
      return parser;
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
  unique_ptr<sdl::SceneParser> scene_parser(parseArguments(argc, argv));

  vector<sdl::Database> dbs;
  vector<sdl::Query> queries;

  // This loads both databases and queries (ie to do end-to-end experiments),
  // but we're only interested in the frames stored in the database
  scene_parser->parseScene(dbs, queries);

  for (int i = dbs.size() - 1; i >= 0; --i) {
    auto& db = dbs[i];
    cout << "Current results will be written to " << db.getCachePath().string()
         << endl;

    sdl::DsoMapGenerator* map_gen =
        static_cast<sdl::DsoMapGenerator*>(db.getMapper());
    map<int, unique_ptr<sdl::Frame>>* frames = db.getFrames();

    // first run the full mapping algorithm
    map_gen->runVisualOdometry();

    // then fetch the depth maps / poses, and convert them to 2D -> 3D image
    // maps
    map<int, cv::SparseMat> scene_coordinate_maps =
        map_gen->getSceneCoordinateMaps();
    map<int, dso::SE3>* poses = map_gen->getPoses();

    for (unsigned int frame_id = 0; frame_id < frames->size(); ++frame_id) {
      auto iter = scene_coordinate_maps.find(frame_id);
      if (iter == scene_coordinate_maps.end() ||
          poses->find(frame_id) == poses->end()) {
        continue;
      } else {
        sdl::Frame* f = frames->at(frame_id).get();
        f->saveSceneCoordinates(iter->second);

        cv::Mat image = f->imageLoader();

        stringstream ss;
        ss << "image_" << setfill('0') << setw(6) << frame_id << ".png";
        cv::imwrite((f->cachePath / ss.str()).string(), image);

        dso::SE3& pose = poses->at(frame_id);
        Eigen::Matrix<double, 3, 4> transform = pose.matrix3x4();

        ss.str("");
        ss << "pose_" << setfill('0') << setw(6) << frame_id << ".bin";
        ofstream pose_file((f->cachePath / ss.str()).string(),
                           ios::binary | ios::trunc);
        pose_file.write(reinterpret_cast<char*>(transform.data()),
                        sizeof(double) * transform.rows() * transform.cols());
      }
    }

    map_gen->saveCameraAdjacencyList(
        (db.getCachePath() / "adjlist.txt").string());

    // pop from the vector, so that old results are erased
    dbs.pop_back();
  }
}
