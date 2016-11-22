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
#ifndef _DFSCUTSGRAPHPARTITIONER_
#define _DFSCUTSGRAPHPARTITIONER_

#include "Partitioner.h"

class DFSCachePartitioner : public Partitioner {
 public:
    DFSCachePartitioner() {};
    ~DFSCachePartitioner() {};

    // Assumptions: datapoints orders (id) are continuous and in order.
    DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads) {
	// Number of datapoints. To be used as graph node id offset.
	int n_datapoints = datapoints.size();

	// Count the number of nodes. This is relatively very fast so it's not a problem
	// that it's taking an extra loop.
	int n_nodes = 0, avg_n_neighbors = 0;
	for (int i = 0; i < datapoints.size(); i++) {
	    int datapoint_id = datapoints[i]->GetOrder() - 1;
	    avg_n_neighbors += datapoints[i]->GetCoordinates().size();
	    for (const auto & coordinate : datapoints[i]->GetCoordinates()) {
		int coordinate_id = coordinate + n_datapoints;
		n_nodes = fmax(n_nodes, coordinate_id);
	    }
	}
	n_nodes++;
	avg_n_neighbors /= n_datapoints;

	// Create the graph.
	std::vector<std::vector<int> > graph(n_nodes+1);
	for (int i = 0; i < graph.size(); i++) {
	    graph[i].reserve(avg_n_neighbors);
	}
	for (int i = 0; i < datapoints.size(); i++) {
	    int datapoint_id = datapoints[i]->GetOrder() - 1;
	    for (const auto & coordinate : datapoints[i]->GetCoordinates()) {
		int coordinate_id = coordinate + n_datapoints;
		graph[datapoint_id].push_back(coordinate_id);
		graph[coordinate_id].push_back(datapoint_id);
	    }
	}

	DatapointPartitions partitions(n_threads);
	int n_points_per_thread = datapoints.size() / n_threads + 1;
	int n_nodes_processed_so_far = 0;

	// Perform dfs on the graph.
	std::vector<int> dfs_stack;
	dfs_stack.reserve(n_nodes);
	std::vector<char> visited(n_nodes);
	std::fill(visited.begin(), visited.end(), 0);
	for (int i = 0; i < datapoints.size(); i++) {
	    dfs_stack.push_back(datapoints[i]->GetOrder()-1);
	}

	int n_datapoints_added = 0;
	while (!dfs_stack.empty()) {
	    int cur_node = dfs_stack[dfs_stack.size()-1];
	    dfs_stack.pop_back();
	    if (visited[cur_node]) {
		continue;
	    }
	    visited[cur_node] = 1;
	    if (cur_node < n_datapoints) {
		int cur_assigned_thread = n_nodes_processed_so_far++ / n_points_per_thread;
		partitions.AddDatapointToThread(datapoints[cur_node], cur_assigned_thread);
		n_datapoints_added++;
	    }
	    for (auto const & neighbor : graph[cur_node]) {
		dfs_stack.push_back(neighbor);
	    }
	}

	if (n_datapoints_added != datapoints.size()) {
  	    std::cout << "DFSCachePartitioner.h: Error, datapoints don't add up - " << n_datapoints_added << " - " << datapoints.size() << std::endl;
	    exit(0);
	}

	return partitions;
    }
};

#endif
