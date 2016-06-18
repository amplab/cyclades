#ifndef _CYCLADES_TRAINER_
#define _CYCLADES_TRAINER_

template<class GRADIENT_CLASS>
class CycladesTrainer : public Trainer<GRADIENT_CLASS> {
public:
    CycladesTrainer() {}
    ~CycladesTrainer() {}

    void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) override {
	CycladesPartitioner partitioner(model);
	Timer partition_timer;
	DatapointPartitions partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}
	Timer gradient_timer;
    }
};

#endif
