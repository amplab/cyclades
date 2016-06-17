#ifndef _HOGWILD_TRAINER_
#define _HOGWILD_TRAINER_

template<class GRADIENT_CLASS>
class HogwildTrainer : public Trainer<GRADIENT_CLASS> {
public:
    HogwildTrainer() {}
    ~HogwildTrainer() {}

    void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) override {
	BasicPartitioner Partitioner;
	DatapointPartitions partitions = Partitioner.Partition(datapoints, FLAGS_n_threads);
	Timer timer;

	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	    if (FLAGS_print_loss_per_epoch) {
		this->PrintTimeLoss(timer, model, datapoints);
	    }
#pragma omp parallel for
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			updater->Update(model, partitions.GetDatapoint(thread, batch, index), thread);
		    }
		}
	    }
	}
    }
};

#endif
