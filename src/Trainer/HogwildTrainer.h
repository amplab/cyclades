#ifndef _HOGWILD_TRAINER_
#define _HOGWILD_TRAINER_

class HogwildTrainer : public Trainer {
public:
    HogwildTrainer() {}
    ~HogwildTrainer() {}

    void Run(Model *model, const std::vector<Datapoint *> & datapoints) override {
	BasicPartitioner Partitioner;
	Partitioner.Partition(datapoints, FLAGS_n_threads);

	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	    if (FLAGS_print_loss_per_epoch) {
		std::cout << model->ComputeLoss(datapoints) << std::endl;
	    }

	}
    }
};

#endif
