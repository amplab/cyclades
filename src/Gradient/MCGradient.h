#ifndef _MCGRADIENT_
#define _MCGRADIENT_

#include "Gradient.h"

class MCGradient : public Gradient {
 public:
    double gradient_coefficient;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	gradient_coefficient = 0;
    }

    MCGradient() {}

    ~MCGradient() {}
};

#endif
