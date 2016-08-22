#ifndef _MATRIXINVERSEDATAPOINT_
#define _MATRIXINVERSEDATAPOINT_

class MatrixInverseDatapoint : public Datapoint {
 private:
    int row;
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
	    input >> index >> weight;
	    coordinates.push_back(index);
	    weights.push_back(weight);
	}
    }

 public:

    MatrixInverseDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~MatrixInverseDatapoint() {}

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
