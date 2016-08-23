#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"

template <class GRADIENT_CLASS>
class Updater {
private:
    Model *model;
    std::vector<Datapoint *> datapoints;
    int n_threads;
    std::vector<int> bookkeeping;

public:
    Updater(Model *model, std::vector<Datapoint *> &datapoints, int n_threads) {
	this->model = model;
	this->datapoints = datapoints;
	for (int i = 0; i < model->NumParameters(); i++) {
	    bookkeeping.push_back(0);
	}
    }
    Updater() {}
    virtual ~Updater() {}

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) = 0;

    virtual void UpdateWrapper(Model *model, Datapoint *datapoint, int thread_num) {
	model->CatchUp(datapoint, datapoint->GetOrder(), bookkeeping);
	Update(model, datapoint, thread_num);
	for (const auto &coordinate : datapoint->GetCoordinates()) {
	    bookkeeping[coordinate] = datapoint->GetOrder();
	}
    }

    // Optional update multiple method.
    virtual void UpdateMultiple(Model *model, DatapointPartitions &partitions, int meta_batch, int thread_num) {
	std::cerr << "Updater: UpdaterMultiple method is not defined." << std::endl;
	exit(0);
    }

    // Called when the epoch ends.
    virtual void EpochFinish() {
	model->EpochFinish();
	for (const auto &datapoint : datapoints) {
	    model->CatchUp(datapoint, model->NumParameters()+1, bookkeeping);
	}
	std::fill(bookkeeping.begin(), bookkeeping.end(), 0);
    }
};

#endif
