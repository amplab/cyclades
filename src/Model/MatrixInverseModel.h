#ifndef _MATRIXINVERSEMODEL_
#define _MATRIXINVERSEMODEL_

#include <sstream>
#include "Model.h"

DEFINE_int32(random_range, 2, "Range of random numbers for initializing the model.");
DEFINE_int32(n_power_iterations, 10, "Number of power iterations to run to calculate lambda.");

class MatrixInverseModel : public Model {
private:
    int n_coords;
    double c_norm;
    double lambda;
    double *model;
    std::vector<double> B;
    std::vector<double> sum_powers;

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
	    for (auto &m_w : ((MatrixInverseDatapoint *)datapoints[dp])->coordinate_weight_map) {
		m_w.second /= norm_factor;
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

	// Precompute sum of powers for "catching up" as this is a dense problem.
	double sum = 0;
	sum_powers.push_back(0);
	for (int i = 0; i < n_coords; i++) {
	    sum += pow(1 - lambda * c_norm * (double)FLAGS_learning_rate, i);
	    sum_powers.push_back(sum);
	}
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
	double sum_sqr = 0, second = 0;
	for (int i = 0; i < n_coords; i++) {
	    second += model[i] * B[i];
	    sum_sqr += model[i] * model[i];
	}

#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:loss)
	for (int i = 0; i < datapoints.size(); i++) {
	    double ai_t_x = 0;
	    double first = sum_sqr * c_norm * lambda;
	    for (int j = 0; j < datapoints[i]->GetWeights().size(); j++) {
		int index = datapoints[i]->GetCoordinates()[j];
		double weight = datapoints[i]->GetWeights()[j];
		ai_t_x += model[index] * weight;
	    }
	    first -= ai_t_x * ai_t_x;
	    loss += first / 2 - second / (double)n_coords;
	}
	return loss;
    }

    void ComputeGradient(Datapoint *datapoint, Gradient *gradient, int thread_num) override {
	MatrixInverseGradient *g = (MatrixInverseGradient *)gradient;
	const std::vector<double> &weights = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	g->gradient_coefficient = 0;
	g->datapoint = datapoint;
	for (int i = 0; i < coordinates.size(); i++) {
	    g->gradient_coefficient += model[coordinates[i]] * weights[i];
	}
	/*MatrixInverseGradient *g = (MatrixInverseGradient *)gradient;
	g->datapoint = datapoint;
	std::map<int, double> m = ((MatrixInverseDatapoint *)datapoint)->coordinate_weight_map;
	double ai_t_x = 0;
	for (int i = 0; i < n_coords; i++) {
	    double weight = 0;
	    if (m.find(i) != m.end()) weight = m[i];
	    ai_t_x += weight * model[i];
	    g->gradient[i] = lambda * c_norm * model[i];
	}
	for (int i = 0; i < n_coords; i++) {
	    double weight = 0;
	    if (m.find(i) != m.end()) weight = m[i];
	    g->gradient[i] -= ai_t_x * weight + B[i] / n_coords;
	    }*/
    }

    void ApplyGradient(Gradient *gradient) override {
	MatrixInverseGradient *g = (MatrixInverseGradient *)gradient;
	Datapoint *datapoint = g->datapoint;
	const std::vector<double> &weights = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	for (int i = 0; i < coordinates.size(); i++) {
	    double complete_gradient = c_norm * lambda * model[coordinates[i]] -
		g->gradient_coefficient * weights[i] -
		B[coordinates[i]] / n_coords;
	    model[coordinates[i]] -= FLAGS_learning_rate * complete_gradient;
	}
	/*MatrixInverseGradient *g = (MatrixInverseGradient *)gradient;
	for (int i = 0; i < n_coords; i++) {
	    model[i] -= FLAGS_learning_rate * g->gradient[i];
	    }*/
    }

    void CatchUp(Datapoint *datapoint, int order, std::vector<int> &bookkeeping) override {
	for (const auto &index : datapoint->GetCoordinates()) {
	    int diff = order - bookkeeping[index] - 1;
	    if (diff < 0) diff = 0;
	    model[index] = model[index] * pow(1 - lambda * c_norm * FLAGS_learning_rate, diff) + sum_powers[diff] * FLAGS_learning_rate * B[index] / n_coords;
	}
    }

    void EpochFinish() override {

    }

    int NumParameters() override {
	return n_coords;
    }
};

#endif
