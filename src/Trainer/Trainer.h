#ifndef _TRAINER_
#define _TRAINER_

class Trainer {
 public:
    Trainer() {}
    virtual ~Trainer() {}

    // Main training method.
    virtual void Train(Model *model, const std::vector<Datapoint *> & datapoints) = 0;
};

#endif
