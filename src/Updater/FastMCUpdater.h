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

#ifndef _FASTMCUPDATER_
#define _FASTMCUPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

// Fast matrix completion SGD updater.
class FastMCSGDUpdater : public SparseSGDUpdater {
protected:

    void PrepareMCGradient(Datapoint *datapoint, Gradient *g) {
	if (g->coeffs.size() != 1) g->coeffs.resize(1);
	const std::vector<double> &labels = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	std::vector<double> &model_data = model->ModelData();
	int rlength = model->CoordinateSize();
	int user_coordinate = coordinates[0];
	int movie_coordinate = coordinates[1];
	double label = labels[0];
	double coeff = 0;
	for (int i = 0; i < rlength; i++) {
	    coeff += model_data[user_coordinate*rlength+i] * model_data[movie_coordinate*rlength+i];
	}
	coeff -= label;
	g->coeffs[0] = coeff;
    }

    void ApplyMCGradient(Datapoint *datapoint, Gradient *gradient) {
	// Custom SGD. This is fast because it avoids intermediate writes to memory,
	// and simply updates the model directly and simultaneously.
	double gradient_coefficient = gradient->coeffs[0];
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	std::vector<double> &model_data = model->ModelData();
	int rlength = model->CoordinateSize();
	int user_coordinate = coordinates[0];
	int movie_coordinate = coordinates[1];
	for (int i = 0; i < rlength; i++) {
	    double new_user_value = model_data[user_coordinate*rlength+i] -
		FLAGS_learning_rate * gradient_coefficient * model_data[movie_coordinate*rlength+i];
	    double new_movie_value = model_data[movie_coordinate*rlength+i] -
		FLAGS_learning_rate * gradient_coefficient * model_data[user_coordinate*rlength+i];
	    model_data[user_coordinate*rlength+i] = new_user_value;
	    model_data[movie_coordinate*rlength+i] = new_movie_value;
	}
    }

    // Note that the Update method is called by many threads.
    // So we have thread local gradients to avoid conflicts.
    void Update(Model *model, Datapoint *datapoint) override {
	int thread_num = omp_get_thread_num();
	thread_gradients[thread_num].Clear();
	thread_gradients[thread_num].datapoint = datapoint;

	// Prepare and apply gradient.
	PrepareMCGradient(datapoint, &thread_gradients[thread_num]);
	ApplyMCGradient(datapoint, &thread_gradients[thread_num]);

	// Update bookkeeping.
	for (const auto &coordinate : datapoint->GetCoordinates()) {
	    bookkeeping[coordinate] = datapoint->GetOrder();
	}
    }

 public:
    FastMCSGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : SparseSGDUpdater(model, datapoints) {
    }

    ~FastMCSGDUpdater() {
    }
};

#endif
