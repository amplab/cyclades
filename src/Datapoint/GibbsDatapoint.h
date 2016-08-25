#ifndef _GIBBSDATAPOINT_
#define _GIBBSDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class GibbsDatapoint : public Datapoint {
private:
    std::vector<double> weights; // Remains empty.
    std::vector<int> neighbors;

    void Initialize(const std::string &input_line) {
	std::stringstream input(input_line);

	// Expect format:
	// Coord# neighbor1 neighbor2 ... neighborn.
	input >> coord;
	while (input) {
	    int neighbor;
	    input >> neighbor;
	    neighbors.push_back(neighbor);
	}
    }

 public:
    int coord;

    GibbsDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }

    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return neighbors;
    }

    int GetNumCoordinateTouches() override {
	return neighbors.size();
    }
};

#endif
