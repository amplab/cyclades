#ifndef _MATRIXINVERSEGRADIENT_
#define _MATRIXINVERSEGRADIENT_

class MatrixInverseGradient : public Gradient {
 public:
    double gradient_coefficient;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	gradient_coefficient = 0;
    }

    void SetUp(Model *model) override {

    }

    MatrixInverseGradient() {}

    ~MatrixInverseGradient() {}
};

#endif
