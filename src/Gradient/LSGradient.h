#ifndef _LSGRADIENT_
#define _LSGRADIENT_

class LSGradient : public Gradient {
 public:
    double gradient_coefficient;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	gradient_coefficient = 0;
    }

    void SetUp(Model *model) override {
	Clear();
    }

    LSGradient() {}

    ~LSGradient() {
    }
};

#endif
