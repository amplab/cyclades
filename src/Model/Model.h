#ifndef _MODEL_
#define _MODEL_

#include "../DatapointPartitions/DatapointPartitions.h"

class Model {
 public:
    Model() {}
    Model(const std::string &input_line) {}
    virtual ~Model() {}

    // Computes loss on the model
    virtual double ComputeLoss(const std::vector<Datapoint *> &datapoints) = 0;

    // Do some set up with the model and datapoints before running gradient descent.
    virtual void SetUp(const std::vector<Datapoint *> &datapoints) {}

    // Do some set up with the model given partitioning scheme before running the trainer.
    virtual void SetUpWithPartitions(DatapointPartitions &partitions) {}

    // Compute and return a gradient. (represented as a void *, so it can be anything).
    // Thread num is the thread number executing the gradient computation.
    virtual void ComputeGradient(Datapoint *, Gradient *gradient, int thread_num) {
	std::cerr << "Model: ComputeGradient is not implemented" << std::endl;
	exit(0);
    }

    // Apply gradient to model.
    virtual void ApplyGradient(Gradient *gradient) {
	std::cerr << "Model: ApplyGradient is not implemented" << std::endl;
	exit(0);
    }

    // The catch-up method to update sparse coordinates before updating
    // them again. For sparse problems this is necessary.
    virtual void CatchUp(Datapoint *datapoint, int datapoint_order, std::vector<int> &bookkeeping) {}

    // Do any sort of extra computation at the end of an epoch.
    virtual void EpochFinish() {}

    // Return the number of parameters of the model.
    virtual int NumParameters() = 0;
};

#endif
