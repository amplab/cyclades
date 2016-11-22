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

#ifndef _CACHE_EFFICIENT_HOGWILD_TRAINER_
#define _CACHE_EFFICIENT_HOGWILD_TRAINER_

#include <map>
#include "../Partitioner/DFSCachePartitioner.h"
#include "../Partitioner/GreedyCachePartitioner.h"

DEFINE_bool(dfs_cache_partitioner, false, "For cache efficient hogwild trainer, use the DFS method to cache partition data points.");
DEFINE_bool(greedy_cache_partitioner, false, "For cache efficient hogwild trainer, use an n^2 greedy algorithm to generate cache friendyl data point ordering.");

class CacheEfficientHogwildTrainer : public Trainer {
protected:
    void PrintStatsAboutProblem(const std::vector<Datapoint *> &datapoints) {
	int n_total_coordinate_accesses = 0;
	int n_distinct_model_accesses = 0;
	double avg_num_coordinates_accessed_per_datapoint = 0;
	int max_coordinates_accessed_per_datapoint = 0;
	int min_coordinates_accessed_per_datapoint = INT_MAX;
	std::map<int, bool> coordinates_set;
	for (int i = 0; i < datapoints.size(); i++) {
	    n_total_coordinate_accesses += datapoints[i]->GetCoordinates().size();
	    for (const auto & coordinate : datapoints[i]->GetCoordinates()) {
		coordinates_set[coordinate] = 1;
	    }
	    max_coordinates_accessed_per_datapoint = fmax(max_coordinates_accessed_per_datapoint, datapoints[i]->GetCoordinates().size());
	    min_coordinates_accessed_per_datapoint = fmin(max_coordinates_accessed_per_datapoint, datapoints[i]->GetCoordinates().size());
	}
	n_distinct_model_accesses = coordinates_set.size();
	avg_num_coordinates_accessed_per_datapoint = n_total_coordinate_accesses / (double)datapoints.size();
	printf("n_datapoints=%d\n"
	       "n_total_coordinate_accesses=%d\n"
	       "n_distinct_model_acceses=%d\n"
	       "avg_num_coordinates_accessed_per_datapoint=%lf\n"
	       "max_coordinates_accessed=%d\n"
	       "min_coordinates_accessed=%d\n",
	       (int)datapoints.size(),
	       n_total_coordinate_accesses,
	       n_distinct_model_accesses,
	       avg_num_coordinates_accessed_per_datapoint,
	       max_coordinates_accessed_per_datapoint,
	       min_coordinates_accessed_per_datapoint);
    }
public:
    CacheEfficientHogwildTrainer() {}
    ~CacheEfficientHogwildTrainer() {}

    TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater *updater) override {
	// Print some stats.
	PrintStatsAboutProblem(datapoints);

	// Partition.
	Timer partition_timer;
	DatapointPartitions partitions(FLAGS_n_threads);
	if (FLAGS_dfs_cache_partitioner) {
	    DFSCachePartitioner partitioner;
	    partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	}
	else if (FLAGS_greedy_cache_partitioner) {
	    GreedyCachePartitioner partitioner;
	    partitions = partitioner.Partition(datapoints, FLAGS_n_threads);
	}
	else {
	    std::cout << "CacheEfficientHogwildTrainer.h: No partitioning method selected" << std::endl;
	    exit(0);
	}
	if (FLAGS_print_partition_time) {
	    this->PrintPartitionTime(partition_timer);
	}

	model->SetUpWithPartitions(partitions);
	updater->SetUpWithPartitions(partitions);

	TrainStatistics stats;

	// Train.
	Timer gradient_timer;
	for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++) {
	    this->EpochBegin(epoch, gradient_timer, model, datapoints, &stats);

	    updater->EpochBegin();

#pragma omp parallel for schedule(static, 1)
	    for (int thread = 0; thread < FLAGS_n_threads; thread++) {
		for (int batch = 0; batch < partitions.NumBatches(); batch++) {
		    for (int index = 0; index < partitions.NumDatapointsInBatch(thread, batch); index++) {
			updater->Update(model, partitions.GetDatapoint(thread, batch, index));
		    }
		}
	    }
	    updater->EpochFinish();
	}
	return stats;
    }
};

#endif
