#ifndef _MATRIXINVERSEMODEL_
#define _MATRIXINVERSEMODEL_

#include <sstream>
#include "Model.h"

DEFINE_int32(random_range, 100, "Range of random numbers for initializing the model.");
DEFINE_int32(n_power_iterations, 10, "Number of power iterations to run to calculate lambda.");

class MatrixInverseModel : public Model {
private:
    int n_coords;
    double c_norm;
    double lambda;
    double *model;
    std::vector<double> B;

    void Initialize(const std::string &input_line) {

	// Input line should have a single number containing
	// number of coordinates (# of rows/columns in square matrix).
	std::stringstream input(input_line);
	input >> n_coords;
	model = (double *)malloc(sizeof(double) * n_coords);

	// Set elements in model to be a random number in range.
	for (int i = 0; i < n_coords; i++) {
	    model[i] = rand() % FLAGS_random_range;
	}
    }

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

	// Do some basic error checking of vector lengths.
	if (temp_vector.size() != output_vector.size() ||
	    temp_vector.size() != n_coords) {
	    std::cerr << "MatrixInverseModel: Wrong size after matrix vector multiply." << std::endl;
	    std::cerr << output_vector.size() << " " << temp_vector.size() << " " << n_coords << std::endl;
	    exit(0);
	}
    }

    void Normalize(std::vector<double> &vec) {
	double norm = 0;
	for (int i = 0; i < vec.size(); i++) {
	    norm += vec[i] * vec[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < vec.size(); i++) {
	    vec[i] /= norm;
	}
    }

    std::vector<Datapoint *> TransposeSparseMatrix(const std::vector<Datapoint *> &d) {
	std::vector<Datapoint *> r;
	for (int i = 0; i < d.size(); i++) {
	    r.push_back(new MatrixInverseDatapoint(std::to_string(i), i));
	}
	for (int row = 0; row < d.size(); row++) {
	    for (int i = 0; i < d[row]->GetWeights().size(); i++) {
		int column_index = d[row]->GetCoordinates()[i];
		double weight = d[row]->GetWeights()[i];
		r[column_index]->GetCoordinates().push_back(row);
		r[column_index]->GetWeights().push_back(weight);
	    }
	}
	return r;
    }

public:
    MatrixInverseModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~MatrixInverseModel() {
	delete model;
    }

    void SetUp(const std::vector<Datapoint *> &datapoints) override {
	// Normalize the rows formed by the datapoint.
	for (int dp = 0; dp < datapoints.size(); dp++) {
	    double sum_sqr = 0;
	    for (const auto &w : datapoints[dp]->GetWeights()) {
		sum_sqr += w*w;
	    }
	    double norm_factor = sqrt(sum_sqr);
	    for (auto &w : datapoints[dp]->GetWeights()) {
		w /= norm_factor;
	    }
	}

	// Calulate Frobenius norm of the matrix.
	double sum_sqr = 0;
	for (int i = 0; i < n_coords; i++) {
	    for (const auto &w : datapoints[i]->GetWeights()) {
		sum_sqr += w*w;
	    }
	}
	c_norm = 1 / sum_sqr;

	// Let B be norm(model^2 * random_vector).
	B.resize(n_coords);

	std::vector<double> random_vector;
	for (int i = 0; i < n_coords; i++) {
	    random_vector.push_back(rand() % FLAGS_random_range);
	}

	MatrixVectorMultiply(datapoints, random_vector, B);
	MatrixVectorMultiply(datapoints, B, B);
	Normalize(B);

	// Calculate lambda via power iteration.
	std::vector<Datapoint *> transpose = TransposeSparseMatrix(datapoints);
	std::vector<double> x_k, x_k_prime;
	for (int i = 0; i < n_coords; i++) {
	    x_k.push_back(rand() % FLAGS_random_range);
	    x_k_prime.push_back(0);
	}
	for (int i = 0; i < FLAGS_n_power_iterations; i++) {
	    MatrixVectorMultiply(datapoints, x_k, x_k);
	    MatrixVectorMultiply(transpose, x_k, x_k);
	    Normalize(x_k);
	}
	MatrixVectorMultiply(datapoints, x_k, x_k_prime);
	MatrixVectorMultiply(transpose, x_k_prime, x_k_prime);
	lambda = 0;
	for (int i = 0; i < n_coords; i++) {
	    lambda += x_k_prime[i] * 1.1 * x_k[i];
	}

	// Free memory of transpose sparse matrix.
	for_each(transpose.begin(), transpose.end(), std::default_delete<Datapoint>());
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
	double sum_sqr = 0, second = 0;
	for (int i = 0; i < n_coords; i++) {
	    second += model[i] * B[i];
	    sum_sqr += model[i] * model[i];
	}
#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:loss)
	for (int i = 0; i < n_coords; i++) {
	    double ai_t_x = 0;
	    double first = sum_sqr * c_norm * lambda;
	    for (int j = 0; j < datapoints[i]->GetWeights().size(); j++) {
		int index = datapoints[i]->GetCoordinates()[j];
		double weight = datapoints[i]->GetWeights()[j];
		ai_t_x += model[index] * weight;
	    }
	    first -= ai_t_x * ai_t_x;
	    loss += first / 2 - second / n_coords;
	}
	return loss;
    }

    void ComputeGradient(Datapoint *datapoint, Gradient *gradient, int thread_num) override {

	return;
    }

    void ApplyGradient(Gradient *gradient) override {
	return;
    }

    int NumParameters() override {
	return n_coords;
    }
};

#endif
