#ifndef _MCDATAPOINT_
#define _MCDATAPOINT_

#include <tuple>

class MCDatapoint : public Datapoint {
 public:
    std::tuple<int, int, double> data;

    MCDatapoint() {}
    ~MCDatapoint() {}

    // Initialize given datapoint given input data line.
    virtual void Initialize(std::string &input_line) {
	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	int user_coord, movie_coord;
	double rating;
	input >> user_coord >> movie_coord >> rating;
	data = std::tuple<int, int, double>(user_coord, movie_coord, rating);
    }
};

#endif
