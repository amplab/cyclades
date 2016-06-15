#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

class CycladesTrainer : public Trainer {
public:
    CycladesTrainer() {}
    ~CycladesTrainer() {}

    void Run(Model *model, const std::vector<Datapoint *> & datapoints) override {
	if (model->ComputeLoss(datapoints) < 100) std::cerr << "WTF?" << std::endl;
    }
};

#endif
