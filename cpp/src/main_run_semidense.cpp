#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>

#include <boost/program_options.hpp>

#include "MapGenerator.h"

namespace po = boost::program_options;

using sdl::DsoMapGenerator;

using namespace std;

const string HELP_FLAG = "help";
const string OUTPUT_PATH_FLAG = "output-path";
const string INPUT_PATH_FLAG = "input-path";
template<typename T>
ostream& operator<<(ostream& out, const vector<T>& vec) {
	bool first = true;
	for (auto element : vec) {
		if (first) {
			out << "[";
			first = false;
		} else {
			out << ", ";
		}
		out << element;
	}
	out << "]";
	return out;
}

int main(int argc, char** argv) {

	// Declare the supported options.
	po::options_description named_args("Allowed options");
	named_args.add_options()(HELP_FLAG.c_str(), "produce help message")(
			OUTPUT_PATH_FLAG.c_str(), po::value<string>(),
			"specify output path");

	po::positional_options_description positional_args;
	po::options_description hidden_args;
	positional_args.add(INPUT_PATH_FLAG.c_str(), -1);
	// boost program_options require positional args to have a corresponding named arg
	// create a separate description and don't print it.
	hidden_args.add_options()(INPUT_PATH_FLAG.c_str(),
			po::value<vector<string>>());

	po::options_description args;
	args.add(named_args).add(hidden_args);

	po::variables_map vm;
	po::store(
			po::command_line_parser(argc, argv).options(args).positional(
					positional_args).run(), vm);
	po::notify(vm);

	if (vm.count(HELP_FLAG) || vm.count(OUTPUT_PATH_FLAG) == 0) {
		cout << "Usage: " << argv[0] << " --" << OUTPUT_PATH_FLAG
				<< "=<output-path> <input-filepath1> <input-filepath2> ..."
				<< endl;
		cout << named_args << "\n";
		return 1;
	}

	string output_path = vm[OUTPUT_PATH_FLAG].as<string>();

	vector<string> input_paths = vm[INPUT_PATH_FLAG].as<vector<string>>();
	int num_inputs_processed = 0;
	for (const auto& input_path : input_paths) {
		// TODO: use overlapping frames, but perturbed enough to produce
		// different pointclouds. ideas: play backwards? drop frames? maybe
		// simple using different start/end indices is sufficient?

		int frames_per_segment = 300;
		unique_ptr<DsoMapGenerator> map_gen(new DsoMapGenerator(input_path));

		map_gen->initVisualOdometry();
		int num_segments = max(1,
				map_gen->getReader().getNumImages() / frames_per_segment);
		stringstream trajectorydir;
		trajectorydir << setfill('0');
		trajectorydir << output_path << "/trajectory_" << setw(5)
				<< num_inputs_processed++;
		for (int segment = 0; segment < num_segments; segment++) {
			vector<int> ids_to_play;
			for (int i = 0; i < frames_per_segment; i++) {
				ids_to_play.push_back(segment * frames_per_segment + i);
			}
			if (segment == num_segments - 1) {
				for (int i = segment * frames_per_segment;
						i < map_gen->getReader().getNumImages(); i++) {
					ids_to_play.push_back(i);
				}
			}

			cout << "running visual odometry with " << ids_to_play.size()
					<< " frames..." << endl;
			map_gen->runVisualOdometry(ids_to_play);

			if (map_gen->hasValidPoints()) {
				stringstream ss;
				ss << setfill('0');
				ss << trajectorydir.str() << "/segment_" << setw(5) << segment
						<< "/";
				map_gen->savePointCloudAsManyPcds(ss.str() + "cloud");
				map_gen->saveDepthMaps(ss.str() + "depth");
				map_gen->saveRawImages(ss.str() + "raw");
				// TODO: save camera calibration. or compute it once over the
				// whole trajectory, then apply that to the short trajectories
			}
		}
		// also run once on full trajectory
		cout << "running visual odometry once on whole trajectory..." << endl;
		map_gen->runVisualOdometry();

		map_gen->savePosesInWorldFrame(input_path + "/groundtruthSync.txt",
				trajectorydir.str() + "/poses.txt");

		map_gen->savePointCloudAsManyPcds(input_path + "/full/cloud");
		map_gen->saveDepthMaps(input_path + "/full/depth");
		map_gen->saveRawImages(input_path + "/full/raw");
	}

	return 0;
}
