#ifndef _BASIC_PARTITIONER_
#define _BASIC_PARTITIONER_

#include "../DatapointPartitions/DatapointPartitions.h"
#include "Partitioner.h"

class BasicPartitioner : public Partitioner {
 public:
    BasicPartitioner() {}
    ~BasicPartitioner() {};

    // Basic partitioner return partition with 1 batch, each thread gets an equal
    // split of a shuffled portion of the datapoints.
    DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads) {

	DatapointPartitions partitions(n_threads);

	// Shuffle the datapoints.
	std::vector<Datapoint *> datapoints_copy(datapoints);
	std::random_shuffle(datapoints_copy.begin(), datapoints_copy.end());

	// Calculate load per thread. Then distribute.
	int n_points_per_thread = datapoints_copy.size() / n_threads;
	for (int thread = 0; thread < n_threads; thread++) {
	    int start = n_points_per_thread * thread;
	    int end = n_points_per_thread * (thread+1);
	    if (thread == n_threads-1) end = datapoints.size();
	    for (int datapoint_count = start; datapoint_count < end; datapoint_count++) {
		partitions.AddDatapointToThread(datapoints_copy[datapoint_count], thread);
	    }
	}

	return partitions;
    }
};

#endif
