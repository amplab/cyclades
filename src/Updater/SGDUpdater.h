#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"

template <class GRADIENT_CLASS>
class SGDUpdater : public Updater<GRADIENT_CLASS> {
private:
    int n_threads;
    GRADIENT_CLASS *thread_gradients;

 public:
    SGDUpdater(int n_threads) : Updater<GRADIENT_CLASS>() {
	thread_gradients = new GRADIENT_CLASS[n_threads];
	this->n_threads = n_threads;
    }

    ~SGDUpdater() {
	delete [] thread_gradients;
    }

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) {
	model->ComputeGradient(datapoint, &thread_gradients[thread_num]);
	model->ApplyGradient(&thread_gradients[thread_num]);
    }
};

#endif
