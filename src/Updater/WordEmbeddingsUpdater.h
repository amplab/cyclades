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

#ifndef _WORDEMBEDDINGSUPDATER_
#define _WORDEMBEDDINGSUPDATER_

#include "Updater.h"
#include "../Gradient/Gradient.h"

// Fast matrix completion SGD updater.
class WordEmbeddingsSGDUpdater : public SparseSGDUpdater {
protected:

    std::vector<double> c_sum_mult1, c_sum_mult2;

    void PrepareWordEmbeddingsGradient(Datapoint *datapoint, Gradient *g) {
	if (g->coeffs.size() != 1) g->coeffs.resize(1);
	const std::vector<double> &labels = datapoint->GetWeights();
	const std::vector<int> &coordinates = datapoint->GetCoordinates();
	int w2v_length = model->CoordinateSize();
	std::vector<double> &local_model = model->ModelData();
	std::vector<double> &C = model->ExtraData();
	int coord1 = coordinates[0];
	int coord2 = coordinates[1];
	double weight = labels[0];
	double norm = 0;
	for (int i = 0; i < w2v_length; i++) {
	    norm += (local_model[coord1*w2v_length+i] + local_model[coord2*w2v_length+i]) *
		(local_model[coord1*w2v_length+i] + local_model[coord2*w2v_length+i]);
	}
	g->coeffs[0] = 2 * weight * (log(weight) - norm - C[0]);

	// Do some extra computation for optimization of C.
	c_sum_mult1[omp_get_thread_num()] += weight * (log(weight) - norm);
	c_sum_mult2[omp_get_thread_num()] += weight;
    }

    void ApplyWordEmbeddingsGradient(Datapoint *datapoint, Gradient *g) {
	int c1 = g->datapoint->GetCoordinates()[0];
	int c2 = g->datapoint->GetCoordinates()[1];
	int w2v_length = model->CoordinateSize();
	std::vector<double> &local_model = model->ModelData();
	for (int i = 0; i < w2v_length; i++) {
	    double final_grad = -(2 * g->coeffs[0] * (local_model[c1*w2v_length+i] + local_model[c2*w2v_length+i]));
	    local_model[c1*w2v_length+i] -= FLAGS_learning_rate * final_grad;
	    local_model[c2*w2v_length+i] -= FLAGS_learning_rate * final_grad;
	}
    }

    // Note that the Update method is called by many threads.
    // So we have thread local gradients to avoid conflicts.
    void Update(Model *model, Datapoint *datapoint) override {
	int thread_num = omp_get_thread_num();
	thread_gradients[thread_num].Clear();
	thread_gradients[thread_num].datapoint = datapoint;

	// Prepare and apply gradient.
	PrepareWordEmbeddingsGradient(datapoint, &thread_gradients[thread_num]);
	ApplyWordEmbeddingsGradient(datapoint, &thread_gradients[thread_num]);

	// Update bookkeeping.
	for (const auto &coordinate : datapoint->GetCoordinates()) {
	    bookkeeping[coordinate] = datapoint->GetOrder();
	}
    }

 public:
    WordEmbeddingsSGDUpdater(Model *model, std::vector<Datapoint *> &datapoints) : SparseSGDUpdater(model, datapoints) {
	c_sum_mult1.resize(FLAGS_n_threads);
	c_sum_mult2.resize(FLAGS_n_threads);
	std::fill(c_sum_mult1.begin(), c_sum_mult1.end(), 0);
	std::fill(c_sum_mult2.begin(), c_sum_mult2.end(), 0);
    }

    ~WordEmbeddingsSGDUpdater() {
    }

    // Called when the epoch ends.
    virtual void EpochFinish() {
	SparseSGDUpdater::EpochFinish();

	// Update C based on closed form solution.
	double C_A = 0, C_B = 0;
	for (int thread = 0; thread < FLAGS_n_threads; thread++) {
	    C_A += c_sum_mult1[thread];
	    C_B += c_sum_mult2[thread];
	    c_sum_mult1[thread] = 0;
	    c_sum_mult2[thread] = 0;
	}
	model->ExtraData()[0] = C_A/C_B;
    }

};

#endif
