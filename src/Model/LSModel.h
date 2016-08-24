#ifndef _LSMODEL_
#define _LSMODEL_

#include <sstream>
#include "Model.h"

class LSModel : public Model {
 private:
    int n_coords;
    double *model;
    double *sum_gradients;
    std::vector<std::map<int, double> > prev_gradients;
    // Artificial label vector.
    std::vector<double> B;

    void MatrixVectorMultiply(const std::vector<Datapoint *> &datapoints,
			      std::vector<double> &input_vector,
			      std::vector<double> &output_vector) {
	// Write to temporary vector to allow for input_vector
	// and output_vector referencing the same vector.
	std::vector<double> temp_vector;

	for (const auto &datapoint : datapoints) {
	    double cross_product = 0;

	    // Each datapoint is like a sparse row in the sparse matrix.
	    for (int i = 0; i < datapoint->GetWeights().size(); i++) {
		int index = datapoint->GetCoordinates()[i];
		double weight = datapoint->GetWeights()[i];
		cross_product += input_vector[index] * weight;
	    }
	    temp_vector.push_back(cross_product);
	}

	// Copy over.
	std::copy(temp_vector.begin(), temp_vector.end(), output_vector.begin());
    }

    void Initialize(const std::string &input_line) {
	// Expect a single number with n_coords.
	std::stringstream input(input_line);
	input >> n_coords;

	// Initialize model.
	model = (double *)malloc(sizeof(double) * n_coords);
	memset(model, 0, sizeof(double) * n_coords);

	// Initialize sum of gradients.
	sum_gradients = (double *)malloc(sizeof(double) * n_coords);
	memset(sum_gradients, 0, sizeof(double) * n_coords);

	// Resize previous gradients.
	prev_gradients.resize(n_coords);
    }
 public:
    LSModel(const std::string &input_line) {
	Initialize(input_line);
    }

    void SetUp(const std::vector<Datapoint *> &datapoints) override {
	B.resize(datapoints.size());

	// Initialize B by multiplying the input matrix with a random vector.
	std::vector<double> rand_vect(n_coords);
	for (int i = 0; i < n_coords; i++) {
	    rand_vect[i] = rand() % FLAGS_random_range;
	}

	MatrixVectorMultiply(datapoints, rand_vect, B);

	// Add some noise to B.
	for (int i = 0; i < datapoints.size(); i++) {
	    B[i] += rand() % FLAGS_random_range;
	}
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:loss)
	for (int i = 0; i < datapoints.size(); i++) {
	    Datapoint *datapoint = datapoints[i];
	    double cross_product = 0;
	    int row = ((LSDatapoint *)datapoint)->row;
	    for (int j = 0; j < datapoint->GetCoordinates().size(); j++) {
		int index = datapoint->GetCoordinates()[j];
		double weight = datapoint->GetWeights()[j];
		cross_product += model[index] * weight;
	    }
	    loss += pow((cross_product - B[row]), 2);
	}
	return loss / datapoints.size();
    }

    void ComputeGradient(Datapoint *datapoint, Gradient *gradient, int thread_num) override {
	LSGradient *lsgrad = (LSGradient *)gradient;
	lsgrad->datapoint = datapoint;
	lsgrad->gradient_coefficient = 0;
	int row = ((LSDatapoint *)datapoint)->row;
	double cp = 0;
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double weight = datapoint->GetWeights()[i];
	    cp += weight * model[index];
	}
	lsgrad->gradient_coefficient = 2 * (cp - B[row]);
    }

    void ApplyGradient(Gradient *gradient) override {
	LSGradient *lsgrad = (LSGradient *)gradient;
	Datapoint *datapoint = lsgrad->datapoint;
	double gradient_coefficient = lsgrad->gradient_coefficient;
	int row = ((LSDatapoint *)datapoint)->row;
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double weight = datapoint->GetWeights()[i];
	    double prev_gradient = prev_gradients[row][index];
	    double sum_gradient = sum_gradients[index];
	    double complete_gradient = (gradient_coefficient*weight) - prev_gradient + sum_gradient / n_coords;
	    model[index] -= FLAGS_learning_rate * complete_gradient;
	    sum_gradients[index] += (gradient_coefficient*weight) - prev_gradient;
	    prev_gradients[row][index] = gradient_coefficient*weight;
	}
    }

    void CatchUp(Datapoint *datapoint, int order, std::vector<int> &bookkeeping) override {
	for (const auto &index : datapoint->GetCoordinates()) {
	    int diff = order - bookkeeping[index] - 1;
	    if (diff < 0) diff = 0;
	    model[index] -= FLAGS_learning_rate * diff * sum_gradients[index] / n_coords;
	}
    }

    int NumParameters() override {
	return n_coords;
    }

    ~LSModel() {
	delete model;
    }
};

#endif
