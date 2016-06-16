#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

class CycladesTrainer : public Trainer {
public:
    CycladesTrainer() {}
    ~CycladesTrainer() {}

    void Train(Model *model, const std::vector<Datapoint *> & datapoints) override {
    }
};

#endif
