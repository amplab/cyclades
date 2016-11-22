/*
* Copyright 2016 [See AUTHORS file for list of authors]
*
*    Licensed under the Apache License, Version 2.0 (the "License");
*    you may not use this file except in compliance with the License.
*    You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS,
*    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*    See the License for the specific language governing permissions and
*    limitations under the License.
*/
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
