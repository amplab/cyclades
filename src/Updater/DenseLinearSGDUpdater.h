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

#ifndef _DENSE_LINEAR_SGD_UPDATER_
#define _DENSE_LINEAR_SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class DenseLinearSGDUpdater : public Updater {
protected:
    REGISTER_THREAD_LOCAL_1D_VECTOR(lambda);
    REGISTER_THREAD_LOCAL_2D_VECTOR(kappa);
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_bar);

    void PrepareNu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &kappa = GET_THREAD_LOCAL_VECTOR(kappa);
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Kappa(index, kappa[index], cur_model);
	}
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<double> &lambda = GET_THREAD_LOCAL_VECTOR(lambda);
	for (int i = 0; i < coordinates.size(); i++) {
	    int index = coordinates[i];
	    model->Lambda(index, lambda[index], cur_model);
	}
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h_bar = GET_THREAD_LOCAL_VECTOR(h_bar);
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H_bar(index, h_bar[index], g, cur_model);
	}
    }

    double H(int coordinate, int index_into_coordinate_vector) {
	return -GET_THREAD_LOCAL_VECTOR(h_bar)[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Nu(int coordinate, int index_into_coordinate_vector) {
	return -GET_THREAD_LOCAL_VECTOR(kappa)[coordinate][index_into_coordinate_vector] * FLAGS_learning_rate;
    }

    double Mu(int coordinate) {
	return GET_THREAD_LOCAL_VECTOR(lambda)[coordinate] * FLAGS_learning_rate;
    }

 public:
    DenseLinearSGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	INITIALIZE_THREAD_LOCAL_1D_VECTOR(lambda, model->NumParameters());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(kappa, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_bar, model->NumParameters(), model->CoordinateSize());
    }

    ~DenseLinearSGDUpdater() {
    }
};

#endif
