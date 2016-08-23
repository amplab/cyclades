#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

template<class GRADIENT_CLASS>
class CycladesTrainer : public Trainer<GRADIENT_CLASS> {
private:
    int *thread_batch;

    // A parallel method where each thread waits
    // until all other threads are on the same batch.
    void WaitForThreadsTilBatch(int thread, int batch) {
	thread_batch[thread] = batch;
	bool waiting_for_threads = true;
	while (waiting_for_threads) {
	    waiting_for_threads = false;
	    for (int t = 0; t < FLAGS_n_threads; t++) {
		if (thread_batch[t] < batch) {
		    waiting_for_threads = true;
		    break;
		}
	    }
	}
    }

    void ClearThreadBatchIndices() {
	memset(thread_batch, 0, sizeof(int) * FLAGS_n_threads);
    }

public:
    CycladesTrainer() {
	thread_batch = new int[FLAGS_n_threads];
	memset(thread_batch, 0, sizeof(int) * FLAGS_n_threads);
    }
    ~CycladesTrainer() {
	delete [] thread_batch;
    }

    void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) override {
	// Partitions.
	CycladesPartitioner partitioner(model);
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
		    WaitForThreadsTilBatch(thread, batch);
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			updater->UpdateWrapper(model, partitions.GetDatapoint(thread, batch, index), thread);
		    }
		}
	    }
	    updater->EpochFinish();
	    ClearThreadBatchIndices();
	}
    }
};

#endif
