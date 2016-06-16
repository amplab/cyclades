#ifndef _MCDATAPOINT_
#define _MCDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class MCDatapoint : public Datapoint {
 private:
    double label;
    int coordinates[2];

    void Initialize(const std::string &input_line) {
	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
    }

 public:

    MCDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~MCDatapoint() {}

    double GetLabel() override {
	return label;
    }

    void * GetData() override {
	return (void *)coordinates;
    }

    int GetNumCoordinateTouches() override {
	return 2;
    }
};

#endif
