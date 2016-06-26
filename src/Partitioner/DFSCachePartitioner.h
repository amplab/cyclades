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

	// Create the graph.
	std::map<int, std::vector<int> > graph;
	for (int i = 0; i < datapoints.size(); i++) {
	    int datapoint_id = datapoints[i]->GetOrder();
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
	std::set<int> visited_nodes;
	dfs_stack.push_back(datapoints[0]->GetOrder());
	while (!dfs_stack.empty()) {
	    int cur_node = dfs_stack[dfs_stack.size()-1];
	    dfs_stack.pop_back();
	    if (visited_nodes.find(cur_node) != visited_nodes.end()) {
		continue;
	    }
	    visited_nodes.insert(cur_node);
	    if (cur_node < n_datapoints) {
		int cur_assigned_thread = n_nodes_processed_so_far++ / n_points_per_thread;
		partitions.AddDatapointToThread(datapoints[cur_node], cur_assigned_thread);
	    }
	    for (auto const & neighbor : graph[cur_node]) {
		dfs_stack.push_back(neighbor);
	    }
	}
	return partitions;
    }
};

#endif
