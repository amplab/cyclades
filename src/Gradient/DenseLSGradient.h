#ifndef _DENSELSGRADIENT_
#define _DENSELSGRADIENT_

#include "Gradient.h"

class DenseLSGradient : public Gradient {
 public:
    Datapoint *datapoint;
    double *gradient;
    int n_params;

    DenseLSGradient() {
	datapoint = NULL;
	gradient = NULL;
	n_params = 0;
    }

    void SetUp(Model *model) override {
	gradient = new double[model->NumParameters()];
	n_params = model->NumParameters();
	Clear();
    }

    void Clear() override {
	memset(gradient, 0, sizeof(double) * n_params);
	datapoint = NULL;
    }

    void Add(const Gradient &other) override {
	const DenseLSGradient &ls_other = dynamic_cast<const DenseLSGradient &>(other);
	datapoint = NULL;
	for (int i = 0; i < n_params; i++) {
	    gradient[i] += ls_other.gradient[i];
	}
    }

    ~DenseLSGradient() {
	if (gradient) {
	    free(gradient);
	}
    }
};

#endif
