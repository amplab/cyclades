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

#ifndef _SPARSE_SGD_UPDATER_
#define _SPARSE_SGD_UPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

class SparseSGDUpdater : public Updater {
protected:
    REGISTER_THREAD_LOCAL_2D_VECTOR(h_bar);

    // Catch up is not required as mu and nu are 0.
    virtual bool NeedCatchUp() {
	return false;
    }

    void PrepareNu(std::vector<int> &coordinates) override {
	// Nu is 0.
    }

    void PrepareMu(std::vector<int> &coordinates) override {
	// Mu is 0.
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
	return 0;
    }

    double Mu(int coordinate) {
	return 0;
    }

 public:
    SparseSGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : Updater(model, datapoints) {
	INITIALIZE_THREAD_LOCAL_2D_VECTOR(h_bar, model->NumParameters(), model->CoordinateSize());
    }

    ~SparseSGDUpdater() {
    }
};

#endif
