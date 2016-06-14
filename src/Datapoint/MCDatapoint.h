#ifndef _MCDATAPOINT_
#define _MCDATAPOINT_

#include <tuple>
#include <sstream>

class MCDatapoint : public Datapoint {
 private:
    double label;
    std::vector<double> coordinates;

 public:

    MCDatapoint() {
	coordinates.resize(2);
    }
    ~MCDatapoint() {}

    // Initialize given datapoint given input data line.
    virtual void Initialize(const std::string &input_line) {
	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
    }

    // Write label to output.
    double GetLabel() override {
	return label;
    }

    // Write data to output.
    const std::vector<double> & GetData() override {
	return coordinates;
    }

    // For matrix completion, each datapoint touches 2 coordinates.
    int GetNumCoordinateTouches() override {
	return coordinates.size();
    }
};

#endif
