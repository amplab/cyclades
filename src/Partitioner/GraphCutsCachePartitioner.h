#ifndef _GRAPHCUTSGRAPHPARTITIONER_
#define _GRAPHCUTSGRAPHPARTITIONER_

#include "Partitioner.h"

class GraphCutsCachePartitioner : public Partitioner {
 public:
    GraphCutsCachePartitioner() {};
    ~GraphCutsCachePartitioner() {};

    DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads) {

    }
}

#endif
