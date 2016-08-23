#ifndef _MATRIXINVERSEGRADIENT_
#define _MATRIXINVERSEGRADIENT_

class MatrixInverseGradient : public Gradient {
 public:
    double gradient_coefficient;
    double *gradient;
    double n_params;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	gradient_coefficient = 0;
	memset(gradient, 0, sizeof(double) * n_params);
    }

    void SetUp(Model *model) override {
	n_params = model->NumParameters();
	gradient = (double *)malloc(sizeof(double) * n_params);
	Clear();
    }

    MatrixInverseGradient() {}

    ~MatrixInverseGradient() {
	delete gradient;
    }
};

#endif
