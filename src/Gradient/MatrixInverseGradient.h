#ifndef _MATRIXINVERSEGRADIENT_
#define _MATRIXINVERSEGRADIENT_

class MatrixInverseGradient : public Gradient {
 public:
    double gradient_coefficient;
    double gradient_coefficient_tilde;
    double n_params;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	gradient_coefficient = 0;
	gradient_coefficient_tilde = 0;
    }

    void SetUp(Model *model) override {
	n_params = model->NumParameters();
	Clear();
    }

    MatrixInverseGradient() {}

    ~MatrixInverseGradient() {
    }
};

#endif
