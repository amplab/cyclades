#ifndef _SGD_UPDATER_
#define _SGD_UPDATER_

#include "Updater.h"

class SGDUpdater : public Updater {
 public:
    SGDUpdater() {}
    ~SGDUpdater() {}

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint) {

    }
};

#endif
