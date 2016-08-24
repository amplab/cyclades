#ifndef _LSDATAPOINT_
#define _LSDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class LSDatapoint : public Datapoint {
 private:
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {

	std::stringstream input(input_line);

	// Expect format:
	// Row# index#1 weight1 index#2 weight2 ...
	input >> row;
	while (input) {
	    int index;
	    double weight;
	    input >> index;
	    if (!input) {
		break;
	    }
	    input >> weight;
	    coordinates.push_back(index);
	    weights.push_back(weight);
	}
    }

 public:
    int row;

    LSDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~LSDatapoint() {}

    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return coordinates.size();
    }
};

#endif
