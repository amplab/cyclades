#ifndef _DENSELSGRADIENT_
#define _DENSELSGRADIENT_

#include "Gradient.h"

class DenseLSGradient : public Gradient {
 public:
    Datapoint *datapoint;
    double gradient_coefficient;

    DenseLSGradient() {}

    void Add(const Gradient &other) {
    }

    ~DenseLSGradient() {}
};

#endif
