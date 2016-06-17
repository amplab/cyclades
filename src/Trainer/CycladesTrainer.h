#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

template<class MODEL_CLASS, class DATAPOINT_CLASS, class GRADIENT_CLASS, class UPDATER_CLASS>
class CycladesTrainer : public Trainer<MODEL_CLASS, DATAPOINT_CLASS, GRADIENT_CLASS, UPDATER_CLASS> {
public:
    CycladesTrainer() {}
    ~CycladesTrainer() {}

    void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) override {
    }
};

#endif
