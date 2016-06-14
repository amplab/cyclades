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

    void Initialize(const std::string &input_line) override {
	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
    }

    double GetLabel() override {
	return label;
    }

    const std::vector<double> & GetData() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return coordinates.size();
    }
};

#endif
