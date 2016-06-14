#ifndef _MODEL_
#define _MODEL_

class Model {
 public:
    Model() {}
    virtual ~Model() {}

    // Initialize model given input line from data file.
    virtual void Initialize(const std::string &input_line) = 0;

    // Computes loss on the model
    virtual double ComputeLoss(const std::vector<Datapoint *> &datapoints) = 0;
};

#endif
