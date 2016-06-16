#ifndef _DATAPOINT_PARTITIONS_
#define _DATAPOINT_PARTITIONS_

class DatapointPartitions {
 private:
    int n_threads;
    std::vector<std::vector<Datapoint *>> datapoints_per_thread;
    std::vector<std::vector<int>> batch_indices;
 public:
    DatapointPartitions(int n_threads) {
	this->n_threads = n_threads;
	datapoints_per_thread.resize(n_threads);
	batch_indices.resize(n_threads);
	for (int i = 0; i < n_threads; i++) {
	    batch_indices[i].push_back(0);
	}
    }
    ~DatapointPartitions() {}

    void start_new_batch() {
	for (int i = 0; i < n_threads; i++) {
	    batch_indices[i].push_back(datapoints_per_thread[i].size());
	}
    }

    void add_datapoint_to_thread(Datapoint * datapoint, int thread) {
	datapoints_per_thread[thread].push_back(datapoint);
    }
};

#endif
