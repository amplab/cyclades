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

#ifndef _SAGA_UPDATER_
#define _SAGA_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SAGAUpdater: public Updater {
 protected:

    // Data structures for capturing the gradient.
    REGISTER_THREAD_LOCAL_2D_VECTOR(h);
    REGISTER_THREAD_LOCAL_DOUBLE(datapoint_order);

    // SAGA data structures.
    REGISTER_GLOBAL_2D_VECTOR(sum_gradients);
    std::vector<std::map<int, std::vector<double> > > prev_gradients;

    void CatchUp(int index, int diff) override {
	if (diff < 0) {
	    diff = 0;
	}
	for (int j = 0; j < model->CoordinateSize(); j++) {
	    model->ModelData()[index*model->CoordinateSize()+j] -=
		FLAGS_learning_rate*diff*GET_GLOBAL_VECTOR(sum_gradients)[index][j] / datapoints.size();
	}
    }

    void PrepareNu(std::vector<int> &coordinates) override {
	// Assuming gradients are sparse, nu should be 0.
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	// We also assume mu is 0.
    }

    void PrepareH(Datapoint *datapoint, Gradient *g) override {
	std::vector<double> &cur_model = model->ModelData();
	std::vector<std::vector<double> > &h = GET_THREAD_LOCAL_VECTOR(h);
	model->PrecomputeCoefficients(datapoint, g, cur_model);
	for (int i = 0; i < datapoint->GetCoordinates().size(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    model->H_bar(index, h[index], g, cur_model);
	}
	GET_THREAD_LOCAL_VECTOR(datapoint_order) = datapoint->GetOrder()-1;
    }

    double H(int coordinate, int index_into_coordinate_vector) override {
	int datapoint_order = GET_THREAD_LOCAL_VECTOR(datapoint_order);
	return FLAGS_learning_rate * (-GET_THREAD_LOCAL_VECTOR(h)[coordinate][index_into_coordinate_vector]
				      + prev_gradients[datapoint_order][coordinate][index_into_coordinate_vector]
				      - GET_GLOBAL_VECTOR(sum_gradients)[coordinate][index_into_coordinate_vector] / datapoints.size());
    }

    double Nu(int coordinate, int index_into_coordinate_vector) override {
	return 0;
    }

    double Mu(int coordinate) override {
	return 0;
    }

    void Update(Model *model, Datapoint *datapoint) {
	Updater::Update(model, datapoint);

	// Update prev and sum gradients.
	int dp_order = datapoint->GetOrder()-1;
	for (const auto &index : datapoint->GetCoordinates()) {
	    for (int i = 0; i < model->CoordinateSize(); i++) {
		GET_GLOBAL_VECTOR(sum_gradients)[index][i] += GET_THREAD_LOCAL_VECTOR(h)[index][i] - prev_gradients[dp_order][index][i];
		prev_gradients[dp_order][index][i] = GET_THREAD_LOCAL_VECTOR(h)[index][i];
	    }
	}
    }

 public:
    SAGAUpdater(Model *model, std::vector<Datapoint *>&datapoints): Updater(model, datapoints) {
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_GLOBAL_2D_VECTOR(sum_gradients, model->NumParameters(), model->CoordinateSize());
	INITIALIZE_THREAD_LOCAL_DOUBLE(datapoint_order);

	// I hope this problem is sparse enough!
	prev_gradients.resize(datapoints.size());
	for (int i = 0; i < datapoints.size(); i++) {
	    int order = datapoints[i]->GetOrder()-1;
	    for (const auto &index : datapoints[i]->GetCoordinates()) {
		prev_gradients[order][index].resize(model->CoordinateSize());
		std::fill(prev_gradients[order][index].begin(),
			  prev_gradients[order][index].end(), 0);
	    }
	}
    }

    ~SAGAUpdater() {}

};

#endif
