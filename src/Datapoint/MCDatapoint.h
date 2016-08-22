#ifndef _MCDATAPOINT_
#define _MCDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class MCDatapoint : public Datapoint {
 private:
    double label;
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {
	// Allocate data for coordiantes / weights.
	coordinates.resize(2);
	weights.resize(2);

	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
	weights[0] = weights[1] = label;
    }

 public:

    MCDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~MCDatapoint() {}

    void OffsetMovieCoord(int offset) {
	coordinates[1] += offset;
    }

    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return 2;
    }
};

#endif
