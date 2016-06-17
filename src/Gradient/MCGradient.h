#ifndef _MCGRADIENT_
#define _MCGRADIENT_

#include "Gradient.h"

class MCGradient : public Gradient {
 public:
    double gradient_coefficient;
    Datapoint *datapoint;

    MCGradient() {}

    MCGradient(Datapoint *datapoint) {
	gradient_coefficient = 0;
	this->datapoint = datapoint;
    }

    ~MCGradient() {}
};

#endif
