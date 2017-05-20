#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "Relocalization.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace sdl;
using namespace cv;

class SceneParser {
public:
	virtual ~SceneParser() {
	}

	/*
	 * Parse a scene into a set of databases and a set of queries (which may be built from overlapping data).
	 */
	virtual void parseScene(vector<sdl::Database>& dbs,
			vector<sdl::Query>& queries) = 0;
};
class SevenScenesParser: public SceneParser {
public:
	SevenScenesParser(const fs::path& directory) :
			directory(directory) {
	}
	virtual ~SevenScenesParser() {
	}
	virtual void parseScene(vector<sdl::Database>& dbs,
			vector<sdl::Query>& queries) {
		vector<fs::path> sequences;
		for (auto dir : fs::recursive_directory_iterator(directory)) {
			if (!fs::is_directory(dir)) {
				continue;
			}
			// each subdirectory should be of the form "seq-XX"
			if (dir.path().filename().string().find("seq") != 0) {
				continue;
			}
			sequences.push_back(dir);
		}
		if (sequences.empty()) {
			throw std::runtime_error("scene contained no sequences!");
		}

		// each sequence can produce 1 database and several queries
		for (const fs::path& sequence_dir : sequences) {
			dbs.push_back(sdl::Database());

			Database& cur_db = dbs.back();
			if (!cache.empty()) {
				cur_db.setCachePath(cache / sequence_dir.filename());
			}

			for (auto file : fs::recursive_directory_iterator(sequence_dir)) {
				string name(file.path().filename().string());
				if (name.find(".color.png") == string::npos) {
					continue;
				}
				// these files are in the format frame-XXXXXX.color.png
				int id = stoi(name.substr(6, 6));
				unique_ptr<Frame> frame(new sdl::Frame(id));
				frame->setImagePath(file);
				if (!cache.empty()) {
					frame->setCachePath(cache / sequence_dir.filename());
				}

				Query q(&cur_db, frame.get());
				queries.push_back(q);
				cur_db.addFrame(move(frame));
			}
		}
	}

	void setCache(fs::path cache_dir){
		cache = cache_dir;
	}

private:
	fs::path directory;
	fs::path cache;
};

unique_ptr<SceneParser> scene_parser;
int vocabulary_size;

void usage(char** argv, const po::options_description commandline_args) {
	cout << "Usage: " << argv[0] << " [options]" << endl;
	cout << commandline_args << "\n";
}

void parseArguments(int argc, char** argv) {
	// Arguments, can be specified on commandline or in a file settings.config
	po::options_description commandline_exclusive(
			"Allowed options from terminal");
	commandline_exclusive.add_options()
			("help", "Print this help message.")
			("config", po::value<string>(), "Path to a config file, which can specify any other argument.");

	po::options_description general_args(
			"Allowed options from terminal or config file");
	general_args.add_options()
			("scene", po::value<string>(), "Type of scene. Currently the only allowed type is 7scenes.")
			("datadir", po::value<string>()->default_value(""), "Directory of the scene dataset. For datasets composed"
					" of several scenes, this should be the appropriate subdirectory.")
			("vocabulary_size", po::value<int>()->default_value(100000), "Size of the visual vocabulary.")
			("cache", po::value<string>()->default_value(""), "Directory to cache intermediate results, ie"
					" descriptors or visual vocabulary, between runs.");

	po::options_description commandline_args;
	commandline_args.add(commandline_exclusive).add(general_args);

	// check for config file
	po::variables_map vm;
	po::store(
			po::command_line_parser(argc, argv).options(commandline_exclusive).allow_unregistered().run(),
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
		// since config is added last, commandline args have precedence over args in the config file.
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
	if(vm.count("cache")){
		cache_dir = fs::path(vm["cache"].as<string>());
	}

	vocabulary_size = vm["vocabulary_size"].as<int>();

	if (vm.count("scene")) {

		string scene_type(vm["scene"].as<string>());
		// currently only one supported scene
		if (scene_type.find("7scenes") == 0) {
			fs::path directory(vm["datadir"].as<string>());
			SevenScenesParser* parser = new SevenScenesParser(directory);
			if(!cache_dir.empty()){
				parser->setCache(cache_dir);
			}
			scene_parser = unique_ptr<SceneParser>(parser);
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

	parseArguments(argc, argv);

	string datadir(argv[1]);
	vector<sdl::Database> dbs;
	vector<sdl::Query> queries;

	scene_parser->parseScene(dbs, queries);

	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	Ptr<BOWImgDescriptorExtractor> bow_extractor = makePtr<
			BOWImgDescriptorExtractor>(sift,
			DescriptorMatcher::create("BruteForce"));

	for (auto& db : dbs) {
		db.setVocabularySize(vocabulary_size);
		db.setDescriptorExtractor(sift);
		db.setFeatureDetector(sift);
		db.setBowExtractor(bow_extractor);
		db.train();
	}
	for (sdl::Query& query : queries) {
		query.setFeatureDetector(sift);
		query.computeFeatures();
	}

	int num_to_return = 10;
	for (unsigned int i = 0; i < dbs.size(); i++) {
		for (unsigned int j = 0; j < queries.size(); j++) {
			if (queries[j].parent_database == &dbs[i]) {
				cout << "testing a query on its original sequence!" << endl;
			}

			vector<sdl::Result> results = dbs[i].lookup(queries[j], num_to_return);

			// compute a homography for each result, then rank by number of inliers

			// TODO: analyzer quality of results somehow

			cout << "keyframes returned: ";
			for (auto result : results) {
				cout << result.frame.index << ", ";
			}
			cout << endl;
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
