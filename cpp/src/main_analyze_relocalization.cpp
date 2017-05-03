#include <iostream>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;
using namespace std;

namespace sdl {

struct Result {

};

class Query {
public:
	Query(const fs::path dir) :
			dataDir(dir) {
	}

	void computeFeatures() {

	}
private:
	const fs::path dataDir;
};

class Database {
public:
	Database(const fs::path dir) :
			dataDir(dir) {
	}

	vector<Result> lookup(const Query& query, int num_to_return) {
		// TODO
		return vector<Result>();
	}

	void train() {
		// TODO
	}
private:
	const fs::path dataDir;
};

void initDatabasesAndQueries(const string& directory, vector<Database>& dbs_out,
		vector<vector<Query>>& queries_out) {

	if (!fs::exists(directory) || !fs::is_directory(directory)) {
		cout << "directory " << directory << " does not exist" << endl;
		exit(-1);
	}

	vector<fs::path> trajectory_paths;
	copy(fs::directory_iterator(directory), fs::directory_iterator(),
			back_inserter(trajectory_paths));
	sort(trajectory_paths.begin(), trajectory_paths.end());

	dbs_out.reserve(trajectory_paths.size());
	queries_out.reserve(trajectory_paths.size());
	for (const auto& trajectory_path : trajectory_paths) {

		dbs_out.emplace_back(trajectory_path / "full");
		vector<Query> curQueries;

		fs::directory_iterator dir_end;
		for (auto subdir = fs::directory_iterator(trajectory_path);
				subdir != dir_end; ++subdir) {
			if (!fs::is_directory(subdir->path())) {
				continue;
			}
			if (subdir->path().filename().compare("full") == 0) {
				continue;
			}
			curQueries.emplace_back(subdir->path());
		}

		queries_out.push_back(curQueries);
	}
}
}

int main(int argc, char** argv) {

	if (argc < 2) {
		cout << "usage: " << argv[0] << " <dataset>" << endl;
		cout << "<dataset> is the output directory of main_run_semidense"
				<< endl;
		return -1;
	}

	string datadir(argv[1]);
	vector<sdl::Database> dbs;
	vector<vector<sdl::Query>> queries;
	sdl::initDatabasesAndQueries(datadir, dbs, queries);

	for (auto& db : dbs) {
		db.train();
	}
	for (vector<sdl::Query>& queryvec : queries) {
		for (auto& query : queryvec) {
			query.computeFeatures();
		}
	}

	int num_to_return = 10;
	for (unsigned int i = 0; i < dbs.size(); i++) {
		for (unsigned int j = 0; j < queries.size(); j++) {
			if (i == j) {
				cout << "testing a query on its original sequence!" << endl;
			}

			for (const sdl::Query& q : queries[j]) {
				vector<sdl::Result> results = dbs[i].lookup(q, num_to_return);
				// TODO: analyzer quality of results somehow
			}

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
