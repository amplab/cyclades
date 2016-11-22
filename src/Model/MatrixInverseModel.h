/*
* Copyright 2016 [See AUTHORS file for list of authors]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/

#ifndef _MATRIXINVERSEMODEL_
#define _MATRIXINVERSEMODEL_

#include <iomanip>
#include <sstream>
#include "Model.h"

DEFINE_int32(n_power_iterations, 10, "Number of power iterations to run to calculate lambda.");

class MatrixInverseModel : public Model {
private:
    int n_coords;
    double lambda;
    std::vector<double> model;
    std::vector<double> B;

    void Initialize(const std::string &input_line) {
	// Input line should have a single number containing
	// number of coordinates (# of rows/columns in square matrix).
	std::stringstream input(input_line);
	input >> n_coords;
	model.resize(n_coords);

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
	for (int i = 0; i < datapoints.size(); i++) {
	    double ai_t_x = 0;
	    double first = sum_sqr / (double)n_coords * lambda;
	    for (int j = 0; j < datapoints[i]->GetWeights().size(); j++) {
		int index = datapoints[i]->GetCoordinates()[j];
		double weight = datapoints[i]->GetWeights()[j];
		ai_t_x += model[index] * weight;
	    }
	    first -= ai_t_x * ai_t_x;
	    loss += first / 2 - second / (double)n_coords;
	}

	return loss + 2;
    }

    int NumParameters() override {
	return n_coords;
    }

    int CoordinateSize() override {
	return 1;
    }

    std::vector<double> & ModelData() override {
	return model;
    }

    void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model) override {
	if (g->coeffs.size() != n_coords) g->coeffs.resize(n_coords);
	const std::vector<double> &weights = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	double product = 0;
	for (int i = 0; i < coordinates.size(); i++) {
	    product += local_model[coordinates[i]] * weights[i];
	}
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    g->coeffs[index] = product * weights[i];
	}
    }

    void Lambda(int coordinate, double &out, std::vector<double> &local_model) override {
	out = lambda / (double)n_coords;
    }

    void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
	out[0] = B[coordinate] / (double)n_coords;
    }

    void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	out[0] = g->coeffs[coordinate];
    }
};

#endif
