#ifndef _WORDEMBEDDINGSDATAPOINT_
#define _WORDEMBEDDINGSDATAPOINT_


class WordEmbeddingsDatapoint : public Datapoint {
 private:
    double label;
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {
	// Allocate data for coordiantes / weights.
	coordinates.resize(2);
	weights.resize(2);

	// Expected input_line format: word_1 index, word_2 index, # of occurrences.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
	weights[0] = weights[1] = label;
    }

 public:

    WordEmbeddingsDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~WordEmbeddingsDatapoint() {}

    const std::vector<double> & GetWeights() override {
	return weights;
    }

    const std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return 2;
    }
};

#endif
