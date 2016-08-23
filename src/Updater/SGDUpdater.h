#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"

template <class GRADIENT_CLASS>
class SGDUpdater : public Updater<GRADIENT_CLASS> {
private:
    int n_threads;
    GRADIENT_CLASS *thread_gradients;

 public:
   SGDUpdater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) : Updater<GRADIENT_CLASS>(model, datapoints, n_threads) {
	thread_gradients = new GRADIENT_CLASS[n_threads];
	this->n_threads = n_threads;
	for (int thread = 0; thread < n_threads; thread++) {
	    thread_gradients[thread] = GRADIENT_CLASS();
	    thread_gradients[thread].SetUp(model);
	}
    }

    ~SGDUpdater() {
	delete [] thread_gradients;
    }

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) {
	thread_gradients[thread_num].Clear();
	model->ComputeGradient(datapoint, &thread_gradients[thread_num], thread_num);
	model->ApplyGradient(&thread_gradients[thread_num]);
    }
};

#endif
