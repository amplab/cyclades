#ifndef _WORDEMBEDDINGSMODEL_
#define _WORDEMBEDDINGSMODEL_

#include <sstream>
#include "Model.h"

DEFINE_int32(vec_length, 30, "Length of word embeddings vector in w2v.");

class WordEmbeddingsModel : public Model {
 private:
    double *model;
    int n_words;
    int w2v_length;

    void InitializePrivateModel() {
	for (int i = 0; i < n_words; i++) {
	    for (int j = 0; j < w2v_length; j++) {
		model[i*w2v_length+j] = ((double)rand()/(double)RAND_MAX);
	    }
	}
    }

    void Initialize(const std::string &input_line) {
	// Expected input_line format: n_words.
	std::stringstream input(input_line);
	input >> n_words;
	w2v_length = FLAGS_vec_length;

	// Allocate memory.
	model = new double[(n_words) * w2v_length];
	if (!model) {
	    std::cerr << "WordEmbeddingsModel: Error allocating model" << std::endl;
	    exit(0);
	}

	// Initialize private model.
	InitializePrivateModel();
    }

 public:
    WordEmbeddingsModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~WordEmbeddingsModel() {
	delete model;
    }

    void SetUp(const std::vector<Datapoint *> &datapoints) override {
	// TODO: add C tracking.
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:loss)
	for (int i = 0; i < datapoints.size(); i++) {
	    Datapoint *datapoint = datapoints[i];
	    const std::vector<double> &labels = datapoint->GetWeights();
	    const std::vector<int> &coordinates = datapoint->GetCoordinates();
	    double weight = labels[0];
	    int x = coordinates[0];
	    int y = coordinates[1];
	    double cross_product = 0;
	    for (int j = 0; j < w2v_length; j++) {
		cross_product += (model[x*w2v_length+j]+model[y*w2v_length+j]) *
		    (model[y*w2v_length+j]+model[y*w2v_length+j]);
	    }
	    loss += weight * (log(weight) - cross_product) * (log(weight) - cross_product);
	}
	return loss / datapoints.size();
    }

    void ComputeGradient(Datapoint * datapoint, Gradient *gradient) override {
	WordEmbeddingsGradient *w2v_gradient = (WordEmbeddingsGradient *)gradient;
	const std::vector<double> &labels = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	int coord1 = coordinates[0];
	int coord2 = coordinates[1];
	double weight = labels[0];
	w2v_gradient->gradient_coefficient = 0;
	w2v_gradient->datapoint = datapoint;
	for (int i = 0; i < w2v_length; i++) {
	    w2v_gradient->gradient_coefficient += (model[coord1*w2v_length+i] + model[coord2*w2v_length+i]) *
		(model[coord1*w2v_length+i] + model[coord2*w2v_length+i]);
	}
	w2v_gradient->gradient_coefficient = 2 * weight * (log(weight) - w2v_gradient->gradient_coefficient);
    }

    void ApplyGradient(Gradient *gradient) override {
	WordEmbeddingsGradient *w2v_gradient = (WordEmbeddingsGradient *)gradient;
	double gradient_coefficient = w2v_gradient->gradient_coefficient;
	Datapoint *datapoint = w2v_gradient->datapoint;
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	int coord1 = coordinates[0];
	int coord2 = coordinates[1];
	for (int i = 0; i < w2v_length; i++) {
	    double gradient = -1 * (gradient_coefficient * 2 * (model[coord1*w2v_length+i] + model[coord2*w2v_length+i]));
	    model[coord1*w2v_length+i] -= FLAGS_learning_rate * gradient;
	    model[coord2*w2v_length+i] -= FLAGS_learning_rate * gradient;
	}
    }

    int NumParameters() override {
	return n_words;
    }
};


#endif
