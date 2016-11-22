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

#ifndef _MCMODEL_
#define _MCMODEL_

#include <sstream>
#include "Model.h"

DEFINE_int32(rlength, 100, "Length of vector in matrix completion.");

class MCModel : public Model {
 private:
    std::vector<double> model;
    int n_users;
    int n_movies;
    int rlength;

    void InitializePrivateModel() {
	for (int i = 0; i < n_users+n_movies; i++) {
	    for (int j = 0; j < rlength; j++) {
		model[i*rlength+j] = ((double)rand()/(double)RAND_MAX);
	    }
	}
    }

    void Initialize(const std::string &input_line) {
	// Expected input_line format: N_USERS N_MOVIES.
	std::stringstream input(input_line);
	input >> n_users >> n_movies;
	rlength = FLAGS_rlength;

	// Allocate memory.
	model.resize((n_users+n_movies) * rlength);

	// Initialize private model.
	InitializePrivateModel();
    }

 public:
    MCModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~MCModel() {
    }

    void SetUp(const std::vector<Datapoint *> &datapoints) override {
	// Update the movies coordinates to reference the second
	// chunk of the model. Do this by offsetting the coordinates
	// by n_users.
	for (const auto & datapoint : datapoints) {
	    ((MCDatapoint *)datapoint)->OffsetMovieCoord(n_users);
	}
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:loss)
	for (int i = 0; i < datapoints.size(); i++) {
	    Datapoint *datapoint = datapoints[i];
	    const std::vector<double> &labels = datapoint->GetWeights();
	    const std::vector<int> &coordinates = datapoint->GetCoordinates();
	    double label = labels[0];
	    int x = coordinates[0];
	    int y = coordinates[1];
	    double cross_product = 0;
	    for (int j = 0; j < rlength; j++) {
		cross_product += model[x*rlength+j] * model[y*rlength+j];
	    }
	    double difference = cross_product - label;
	    loss += difference * difference;
	}
	return loss / datapoints.size();
    }

    std::vector<double> & ModelData() {
	return model;
    }

    int NumParameters() override {
	return n_users + n_movies;
    }

    int CoordinateSize() override {
	return rlength;
    }

    void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model) override {
	if (g->coeffs.size() != 1) g->coeffs.resize(1);
	const std::vector<double> &labels = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	int user_coordinate = coordinates[0];
	int movie_coordinate = coordinates[1];
	double label = labels[0];
	double coeff = 0;
	for (int i = 0; i < rlength; i++) {
	    coeff += local_model[user_coordinate*rlength+i] * local_model[movie_coordinate*rlength+i];
	}
	coeff -= label;
	g->coeffs[0] = coeff;
    }

    void Lambda(int coordinate, double &out, std::vector<double> &local_model) override {
    }

    void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
    }

    void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	double other_coordinate = 0;
	if (g->datapoint->GetCoordinates()[0] == coordinate)
	    other_coordinate = g->datapoint->GetCoordinates()[1];
	else
	    other_coordinate = g->datapoint->GetCoordinates()[0];
	for (int i = 0; i < rlength; i++) {
	    out[i] = g->coeffs[0] * local_model[other_coordinate * rlength + i];
	}
    }
};

#endif
