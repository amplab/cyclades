#ifndef _DENSELSGRADIENT_
#define _DENSELSGRADIENT_

#include "Gradient.h"

class DenseLSGradient : public Gradient {
 public:
    Datapoint *datapoint;
    double gradient_coefficient;

    DenseLSGradient() {}

    void Clear() override {
	gradient_coefficient = 0;
	datapoint = NULL;
    }

    void Add(const Gradient &other) override {
	const DenseLSGradient &ls_other = dynamic_cast<const DenseLSGradient &>(other);
	datapoint = ls_other.datapoint;
	gradient_coefficient = ls_other.gradient_coefficient;
    }

    ~DenseLSGradient() {}
};

#endif
