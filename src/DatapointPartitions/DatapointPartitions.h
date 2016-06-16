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

    void StartNewBatch() {
	for (int i = 0; i < n_threads; i++) {
	    batch_indices[i].push_back(datapoints_per_thread[i].size());
	}
    }

    void AddDatapointToThread(Datapoint * datapoint, int thread) {
	datapoints_per_thread[thread].push_back(datapoint);
    }

    int NumBatches() {
	return batch_indices[0].size();
    }

    int NumDatapointsInBatch(int thread, int batch) {
	// Last batch.
	if (batch == NumBatches()-1) {
	    return datapoints_per_thread[thread].size() - batch_indices[thread][batch];
	}
	return batch_indices[thread][batch+1] - batch_indices[thread][batch];
    }

    Datapoint * GetDatapoint(int thread, int batch, int index) {
	int real_index = batch_indices[thread][batch] + index;
	return datapoints_per_thread[thread][real_index];
    }
};

#endif
