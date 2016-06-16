#ifndef _MCDATAPOINT_
#define _MCDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class MCDatapoint : public Datapoint {
 private:
    double label;
    std::vector<double> labels;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {
	// Allocate data for coordiantes / labels.
	coordinates.resize(2);
	labels.resize(2);

	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
	labels[0] = labels[1] = label;
    }

 public:

    MCDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~MCDatapoint() {}

    const std::vector<double> & GetLabels() override {
	return labels;
    }

    const std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return 2;
    }
};

#endif
