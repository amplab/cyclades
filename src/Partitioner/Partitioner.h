#ifndef _PARTITIONER_
#define _PARTITIONER_

#include "../DatapointPartitions/DatapointPartitions.h"

class Partitioner {
public:
    Partitioner() {}
    virtual ~Partitioner() {}

    // Main partitioning method. Partitions a vector of
    // Datapoint * given number of threads.
    // Return value of form [batch][thread][datapoint pointers].
    virtual DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads) = 0;
};

#endif
