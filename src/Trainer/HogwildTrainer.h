#ifndef _HOGWILD_TRAINER_
#define _HOGWILD_TRAINER_

class HogwildTrainer : public Trainer {
public:
    HogwildTrainer() {}
    ~HogwildTrainer() {}

    void Run(Model *model, const std::vector<Datapoint *> & datapoints) override {
    }
};

#endif
