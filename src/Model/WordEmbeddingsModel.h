#ifndef _WORDEMBEDDINGSMODEL_
#define _WORDEMBEDDINGSMODEL_

#include <sstream>
#include "../DatapointPartitions/DatapointPartitions.h"
#include "Model.h"

DEFINE_int32(vec_length, 30, "Length of word embeddings vector in w2v.");

class WordEmbeddingsModel : public Model {
 private:
    double *model;
    double C;
    std::vector<std::vector<double> > c_sum_mult1, c_sum_mult2;
    std::vector<double> c_thread_index_tracker;
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

	C = 0;

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

    void SetUpWithPartitions(DatapointPartitions &partitions) override {
	// Initialize C_sum_mult variables.
	c_sum_mult1.resize(FLAGS_n_threads);
	c_sum_mult2.resize(FLAGS_n_threads);
	c_thread_index_tracker.resize(FLAGS_n_threads);
	// First calculate number of datapoints per thread.
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    int n_datapoints_for_thread = 0;
	    for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		n_datapoints_for_thread += partitions.NumDatapointsInBatch(thread, batch);
	    }
	    c_sum_mult1[thread].resize(n_datapoints_for_thread);
	    c_sum_mult2[thread].resize(n_datapoints_for_thread);
	}
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
	    loss += weight * (log(weight) - cross_product - C) * (log(weight) - cross_product - C);
	}
	return loss / datapoints.size();
    }

    void ComputeGradient(Datapoint * datapoint, Gradient *gradient, int thread_num) override {
	WordEmbeddingsGradient *w2v_gradient = (WordEmbeddingsGradient *)gradient;
	const std::vector<double> &labels = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	int coord1 = coordinates[0];
	int coord2 = coordinates[1];
	double weight = labels[0];
	w2v_gradient->gradient_coefficient = 0;
	w2v_gradient->datapoint = datapoint;
	double norm = 0;
	for (int i = 0; i < w2v_length; i++) {
	    norm += (model[coord1*w2v_length+i] + model[coord2*w2v_length+i]) *
		(model[coord1*w2v_length+i] + model[coord2*w2v_length+i]);
	}
	w2v_gradient->gradient_coefficient = 2 * weight * (log(weight) - norm - C);

	// Update c_sum_mults to calculate C.
	int index = c_thread_index_tracker[thread_num]++;
	c_sum_mult1[thread_num][index] = weight * (log(weight) - norm);
	c_sum_mult2[thread_num][index] = weight;
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

    void EpochFinish() {
	// Update C based on C_sum_mult.
	double C_A = 0, C_B = 0;
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    for (int index = 0; index < c_sum_mult1[thread].size(); index++) {
		C_A += c_sum_mult1[thread][index];
		C_B += c_sum_mult2[thread][index];
	    }
	}
	C = C_A / C_B;

	// Reset c sum index tracker.
	std::fill(c_thread_index_tracker.begin(), c_thread_index_tracker.end(), 0);
    }

    int NumParameters() override {
	return n_words;
    }
};


#endif
