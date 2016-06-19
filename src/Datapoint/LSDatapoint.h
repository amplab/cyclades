#ifndef _LSDATAPOINT_
#define _LSDATAPOINT_

#include "Datapoint.h"

class LSDatapoint : public Datapoint {
private:
    double label;
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {
	// Expect inputs of format: correct_label coordinate_1 weight_1 coordinate_2 weight_2...
	std::stringstream input(input_line);
	int cur_coordinate;
	double cur_label;
	input >> label;
	while (input) {
	    input >> cur_coordinate >> cur_label;
	    weights.push_back(cur_label);
	    coordinates.push_back(cur_coordinate);
	}
    }

public:
    LSDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }

    ~LSDatapoint() {}

    const std::vector<double> & GetWeights() {
	return weights;
    }

    const std::vector<int> & GetCoordinates() {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return coordinates.size();
    }

    int GetLabel() {
	return label;
    }
};

#endif
