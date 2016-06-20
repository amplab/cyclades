#ifndef _UPDATER_
#define _UPDATER_

#include "../DatapointPartitions/DatapointPartitions.h"

template <class GRADIENT_CLASS>
class Updater {
 public:
    Updater(int n_threads) {}
    Updater() {}
    virtual ~Updater() {}

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) = 0;

    // Optional update multiple method.
    virtual void UpdateMultiple(Model *model, DatapointPartitions &partitions, int meta_batch, int thread_num) {
	std::cerr << "Updater: UpdaterMultiple method is not defined." << std::endl;
	exit(0);
    }

};

#endif
