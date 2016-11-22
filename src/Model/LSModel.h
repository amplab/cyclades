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

#ifndef _LSMODEL_
#define _LSMODEL_

#include <sstream>
#include "Model.h"

class LSModel : public Model {
 private:
    int n_coords;
    std::vector<double> model;
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
	model.resize(n_coords);
	std::fill(model.begin(), model.end(), 0);
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
	    rand_vect[i] = (rand() % FLAGS_random_range);
	}

	MatrixVectorMultiply(datapoints, rand_vect, B);

	// Add some noise to B.
	for (int i = 0; i < datapoints.size(); i++) {
	    //B[i] += rand() % FLAGS_random_range;
	}
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;

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
	int row = ((LSDatapoint *)datapoint)->row;
	double cp = 0;
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double weight = datapoint->GetWeights()[i];
	    cp += weight * local_model[index];
	}
	double partial_grad = 2 * (cp - B[row]);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double weight = datapoint->GetWeights()[i];
	    g->coeffs[index] = partial_grad * weight;
	}
    }

    void Lambda(int coordinate, double &out, std::vector<double> &local_model) override {
	out = 0;
    }

    void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
	out[0] = 0;
    }

    void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	out[0] = g->coeffs[coordinate];
    }

    ~LSModel() {
    }
};

#endif
