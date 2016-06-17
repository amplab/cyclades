#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

template<class GRADIENT_CLASS>
class CycladesTrainer : public Trainer<GRADIENT_CLASS> {
public:
    CycladesTrainer() {}
    ~CycladesTrainer() {}

    void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) override {
    }
};

#endif
