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
};

#endif
