#ifndef _TRAINER_
#define _TRAINER_

class Trainer {
 public:
    Trainer() {}
    virtual ~Trainer() {}

    // Training method.
    virtual void Run(Model *model, const std::vector<Datapoint *> & datapoints) = 0;
};

#endif
