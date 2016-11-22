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

#ifndef _SVRG_UPDATER_
#define _SVRG_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SVRGUpdater : public Updater {
protected:
    double n_updates_so_far;

    std::vector<double> model_copy;
    // Vectors for computing SVRG related data.
    REGISTER_THREAD_LOCAL_1D_VECTOR(lambda);
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_x);
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_y);
    REGISTER_GLOBAL_1D_VECTOR(g);

    // Vectors for computing the sum of gradients (g).
    REGISTER_THREAD_LOCAL_2D_VECTOR(g_kappa);
    REGISTER_THREAD_LOCAL_1D_VECTOR(g_lambda);
    REGISTER_THREAD_LOCAL_2D_VECTOR(g_h_bar);
    REGISTER_GLOBAL_1D_VECTOR(n_zeroes);

    void PrepareMu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &lambda = GET_THREAD_LOCAL_VECTOR(lambda);
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Lambda(index, lambda[index], cur_model);
	}
    }

    void PrepareNu(std::vector<int> &coordinates) override {
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h_x = GET_THREAD_LOCAL_VECTOR(h_x);
	std::vector<std::vector<double> > &h_y = GET_THREAD_LOCAL_VECTOR(h_y);

	g->datapoint = datapoint;
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	int coord_size = model->CoordinateSize();
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H_bar(index, h_x[index], g, cur_model);
	}
	model->PrecomputeCoefficients(datapoint, g, model_copy);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H_bar(index, h_y[index], g, model_copy);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector) {
	return -FLAGS_learning_rate * (GET_THREAD_LOCAL_VECTOR(h_x)[coordinate][index_into_coordinate_vector] -
				       GET_THREAD_LOCAL_VECTOR(h_y)[coordinate][index_into_coordinate_vector]);
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return FLAGS_learning_rate * (GET_GLOBAL_VECTOR(g)[coordinate*model->CoordinateSize()+index_into_coordinate_vector] -
				      GET_THREAD_LOCAL_VECTOR(lambda)[coordinate] * model_copy[coordinate*model->CoordinateSize()+index_into_coordinate_vector]);
    }

    double Mu(int coordinate) {
	return GET_THREAD_LOCAL_VECTOR(lambda)[coordinate] * FLAGS_learning_rate;
    }

    void ModelCopy() {

	// Make a copy of the model every epoch.
	model_copy = model->ModelData();

	// Clear the sum of gradients.
	std::vector<double> &g = GET_GLOBAL_VECTOR(g);
	std::fill(g.begin(), g.end(), 0);

	// Compute average sum of gradients on the model copy.
	std::vector<double> &n_zeroes = GET_GLOBAL_VECTOR(n_zeroes);
	int coord_size = model->CoordinateSize();

	// zero gradients.
#pragma omp parallel for num_threads(FLAGS_n_threads)
	for (int coordinate = 0; coordinate < model->NumParameters(); coordinate++) {
	    std::vector<std::vector<double> > &g_kappa = GET_THREAD_LOCAL_VECTOR(g_kappa);
	    std::vector<double> &g_lambda = GET_THREAD_LOCAL_VECTOR(g_lambda);
	    model->Kappa(coordinate, g_kappa[coordinate], model_copy);
	    model->Lambda(coordinate, g_lambda[coordinate], model_copy);
	    for (int j = 0; j < coord_size; j++) {
		g[coordinate*coord_size+j] = (g_lambda[coordinate] * model_copy[coordinate*coord_size+j] - g_kappa[coordinate][j]) * n_zeroes[coordinate];
	    }
	}

	// non zero gradients. Essentially do SGD here, on the same partitioning pattern.
#pragma omp parallel num_threads(FLAGS_n_threads)
	{
	    int thread = omp_get_thread_num();
	    for (int batch = 0; batch < datapoint_partitions->NumBatches(); batch++) {
#pragma omp barrier
		for (int index = 0; index < datapoint_partitions->NumDatapointsInBatch(thread, batch); index++) {
		    Datapoint *datapoint = datapoint_partitions->GetDatapoint(thread, batch, index);
		    Gradient *grad = &thread_gradients[omp_get_thread_num()];
		    grad->datapoint = datapoint;
		    model->PrecomputeCoefficients(datapoint, grad, model_copy);
		    std::vector<std::vector<double> > &g_kappa = GET_THREAD_LOCAL_VECTOR(g_kappa);
		    std::vector<double> &g_lambda = GET_THREAD_LOCAL_VECTOR(g_lambda);
		    std::vector<std::vector<double> > &g_h_bar = GET_THREAD_LOCAL_VECTOR(g_h_bar);
		    for (auto & coord : datapoint->GetCoordinates()) {
			model->H_bar(coord, g_h_bar[coord], grad, model_copy);
			model->Lambda(coord, g_lambda[coord], model_copy);
			model->Kappa(coord, g_kappa[coord], model_copy);
		    }
		    for (auto & coord : datapoint->GetCoordinates()) {
			for (int j = 0; j < coord_size; j++) {
			    g[coord*coord_size+j] += g_lambda[coord] * model_copy[coord*coord_size+j]
				- g_kappa[coord][j] + g_h_bar[coord][j];
			}
		    }
		}
	    }
	}

#pragma omp parallel for num_threads(FLAGS_n_threads)
	for (int i = 0; i < model->NumParameters(); i++) {
	    for (int j = 0; j < coord_size; j++) {
		g[i*coord_size+j] /= datapoints.size();
	    }
	}
    }

 public:
 SVRGUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	INITIALIZE_GLOBAL_1D_VECTOR(g, model->NumParameters() * model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_1D_VECTOR(lambda, model->NumParameters());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_x, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_y, model->NumParameters(), model->CoordinateSize());
	model_copy.resize(model->ModelData().size());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(g_kappa, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_1D_VECTOR(g_lambda, model->NumParameters());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(g_h_bar, model->NumParameters(), model->CoordinateSize());

	// Compute number of zeroes for each column (parameters) of the model.
	INITIALIZE_GLOBAL_1D_VECTOR(n_zeroes, model->NumParameters());
	std::vector<double> &n_zeroes = GET_GLOBAL_VECTOR(n_zeroes);
	for (int i = 0; i < model->NumParameters(); i++) {
	    n_zeroes[i] = datapoints.size();
	}
	int sum = 0;
	for (int dp = 0; dp < datapoints.size(); dp++) {
	    for (auto &coordinate : datapoints[dp]->GetCoordinates()) {
		n_zeroes[coordinate]--;
		sum++;
	    }
	}
    }

    void Update(Model *model, Datapoint *datapoint) override {
	Updater::Update(model, datapoint);
    }

    void EpochBegin() override {
	Updater::EpochBegin();
	ModelCopy();
    }

    ~SVRGUpdater() {
    }
};

#endif
