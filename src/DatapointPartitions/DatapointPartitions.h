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

#ifndef _DATAPOINT_PARTITIONS_
#define _DATAPOINT_PARTITIONS_

typedef std::tuple<int, int> ThreadLoadPair;
struct ThreadLoadComp {
    bool operator()(const ThreadLoadPair &s1, const ThreadLoadPair &s2) {
	return std::get<1>(s1) > std::get<1>(s2);
    }
};


class DatapointPartitions {
 private:
    int n_threads;
    std::vector<std::vector<Datapoint *>> datapoints_per_thread;
    std::vector<std::vector<int>> batch_indices;
    std::vector<ThreadLoadPair> thread_load_heap;

    void ClearThreadLoadHeap() {
	for (int i = 0; i < n_threads;i ++) {
	    std::get<0>(thread_load_heap[i]) = i;
	    std::get<1>(thread_load_heap[i]) = 0;
	}
    }

 public:
    DatapointPartitions(int n_threads) {
	this->n_threads = n_threads;
	datapoints_per_thread.resize(n_threads);
	batch_indices.resize(n_threads);
	for (int i = 0; i < n_threads; i++) {
	    batch_indices[i].push_back(0);
	}
	thread_load_heap.resize(n_threads);
	ClearThreadLoadHeap();
    }
    ~DatapointPartitions() {}

    void StartNewBatch() {
	for (int i = 0; i < n_threads; i++) {
	    batch_indices[i].push_back(datapoints_per_thread[i].size());
	}
	ClearThreadLoadHeap();
    }

    void AddDatapointToThread(Datapoint * datapoint, int thread) {
	datapoints_per_thread[thread].push_back(datapoint);
    }

    void AddDatapointsToLeastLoadedThread(const std::vector<Datapoint *> &datapoints) {
	// Get least loaded thread.
	ThreadLoadPair lightest_thread_load_pair = thread_load_heap.front();
	int lightest_thread = std::get<0>(lightest_thread_load_pair);
	int weight = std::get<1>(lightest_thread_load_pair);

	// Remove lightest thread-load pair.
	std::pop_heap(thread_load_heap.begin(),
		      thread_load_heap.end(),
		      ThreadLoadComp());
	thread_load_heap.pop_back();

	// Add.
	for (auto const & datapoint : datapoints) {
	    AddDatapointToThread(datapoint, lightest_thread);
	}

	// Add the updated thread-load pair back to the heap
	std::get<1>(lightest_thread_load_pair) = weight+datapoints.size();
	thread_load_heap.push_back(lightest_thread_load_pair);
	std::push_heap(thread_load_heap.begin(),
		       thread_load_heap.end(),
		       ThreadLoadComp());
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
