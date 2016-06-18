#ifndef _MODEL_
#define _MODEL_

class Model {
 public:
    Model() {}
    Model(const std::string &input_line) {}
    virtual ~Model() {}

    // Computes loss on the model
    virtual double ComputeLoss(const std::vector<Datapoint *> &datapoints) = 0;

    // Do some set up with the model and datapoints before running gradient descent.
    virtual void SetUp(const std::vector<Datapoint *> &datapoints) {}

    // Compute and return a gradient. (represented as a void *, so it can be anything).
    virtual void ComputeGradient(Datapoint *, Gradient *gradient) {
	std::cerr << "Model: ComputeGradient is not implemented" << std::endl;
	exit(0);
    }

    // Apply gradient to model.
    virtual void ApplyGradient(Gradient *gradient) {
	std::cerr << "Model: ApplyGradient is not implemented" << std::endl;
	exit(0);
    }

    // Return the number of parameters of the model.
    virtual int NumParameters() = 0;
};

#endif
