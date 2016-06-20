#ifndef _DENSELSDATAPOINT_
#define _DENSELSDATAPOINT_

#include "Datapoint.h"

class DenseLSDatapoint : public Datapoint {
private:
    double label;
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {
	// Expect inputs of format: correct_label coordinate_1 weight_1 coordinate_2 weight_2...
	std::string input_line_copy = input_line;
	input_line_copy.erase(std::remove(input_line_copy.begin(), input_line_copy.end(), '\n'), input_line_copy.end());
	std::stringstream input(input_line_copy);
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
    DenseLSDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }

    ~DenseLSDatapoint() {}

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
