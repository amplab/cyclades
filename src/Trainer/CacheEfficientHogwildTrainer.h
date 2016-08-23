#ifndef _CACHE_EFFICIENT_HOGWILD_TRAINER_
#define _CACHE_EFFICIENT_HOGWILD_TRAINER_

template<class GRADIENT_CLASS>
class CacheEfficientHogwildTrainer : public Trainer<GRADIENT_CLASS> {
public:
    CacheEfficientHogwildTrainer() {}
    ~CacheEfficientHogwildTrainer() {}

    void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) override {
	// Partition.
	DFSCachePartitioner partitioner;
	Timer partition_timer;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}

	model->SetUpWithPartitions(partitions);

	// Train.
	Timer gradient_timer;
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	    if (FLAGS_print_loss_per_epoch) {
		this->PrintTimeLoss(gradient_timer, model, datapoints);
	    }
#pragma omp parallel for schedule(static, 1)
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			updater->UpdateWrapper(model, partitions.GetDatapoint(thread, batch, index), thread);
		    }
		}
	    }
	    updater->EpochFinish();
	}
    }
};

#endif
