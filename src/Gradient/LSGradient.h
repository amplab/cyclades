#ifndef _LSGRADIENT_
#define _LSGRADIENT_

#include "Gradient.h"

class LSGradient : public Gradient {
 public:
    Datapoint *datapoint;
    double gradient_coefficient;

    LSGradient() {}

    LSGradient(Datapoint *datapoint) {
	gradient_coefficient = 0;
	this->datapoint = datapoint;
    }

    ~LSGradient() {}
};

#endif
