#ifndef _GIBBSGRADIENT_
#define _GIBBSGRADIENT_

#include "Gradient.h"

class GibbsGradient : public Gradient {
 public:
    Datapoint *datapoint;

    void SetUp(Model *model) override {
	// Do nothing.
    }

    GibbsGradient() {
	datapoint = NULL;
    }

    void Clear() {
	datapoint = NULL;
    }
};

#endif
