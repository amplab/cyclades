
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
#ifndef _DEFINES_
#define _DEFINES_

#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <thread>
#include <map>
#include <set>
#include <cstring>
#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include "Datapoint/Datapoint.h"
#include "Gradient/Gradient.h"
#include "DatasetReader.h"
#include "DatapointPartitions/DatapointPartitions.h"
#include "Partitioner/Partitioner.h"
#include "Partitioner/BasicPartitioner.h"
#include "Model/Model.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <sys/time.h>
int clock_gettime(int /*clk_id*/, struct timespec* t) {
    struct timeval now;
    int rv = gettimeofday(&now, NULL);
    if (rv) return rv;
    t->tv_sec  = now.tv_sec;
    t->tv_nsec = now.tv_usec * 1000;
    return 0;
}
#define CLOCK_MONOTONIC 0
#endif

class Timer {
public:
    struct timespec _start;
    struct timespec _end;
    Timer(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    virtual ~Timer(){}
    inline void Restart(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    inline float Elapsed(){
        clock_gettime(CLOCK_MONOTONIC, &_end);
        return (_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1000000000.0;
    }
};

void pin_to_core(size_t core) {
#ifdef _GNU_SOURCE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

DEFINE_string(data_file, "blank", "Input data file.");
DEFINE_int32(n_epochs, 100, "Number of passes of data in training.");
DEFINE_int32(n_threads, 2, "Number of threads in parallel during training.");
DEFINE_double(learning_rate, .001, "Learning rate.");
DEFINE_bool(print_loss_per_epoch, false, "Should compute and print loss every epoch.");
DEFINE_bool(print_partition_time, false, "Should print time taken to distribute datapoints across threads.");


DEFINE_bool(shuffle_datapoints, true, "Shuffle datapoints before training.");

// Flags for training types.
DEFINE_bool(cache_efficient_hogwild_trainer, false, "Hogwild training method with cache friendly datapoint ordering (parallel).");
DEFINE_bool(cyclades_trainer, false, "Cyclades training method (parallel).");
DEFINE_bool(hogwild_trainer, false, "Hogwild training method (parallel).");

// Flags for updating types.
DEFINE_bool(dense_linear_sgd, false, "Use the dense linear SGD update method.");
DEFINE_bool(sparse_sgd, false, "Use the sparse SGD update method.");
DEFINE_bool(svrg, false, "Use the SVRG update method.");
DEFINE_bool(saga, false, "Use the SAGA update method. Note this assumes gradients are sparse.");

// MISC flags.
DEFINE_int32(random_range, 100, "Range of random numbers for initializing the model.");

#include "Updater/Updater.h"
#include "Updater/DenseLinearSGDUpdater.h"
#include "Updater/SparseSGDUpdater.h"
#include "Updater/SVRGUpdater.h"
#include "Updater/SAGAUpdater.h"
#include "Updater/FastMCUpdater.h"
#include "Updater/WordEmbeddingsUpdater.h"

#include "Partitioner/CycladesPartitioner.h"
#include "Partitioner/DFSCachePartitioner.h"
#include "Trainer/Trainer.h"
#include "Trainer/CycladesTrainer.h"
#include "Trainer/HogwildTrainer.h"
#include "Trainer/CacheEfficientHogwildTrainer.h"

#include "Datapoint/MCDatapoint.h"
#include "Model/MCModel.h"

#include "Datapoint/WordEmbeddingsDatapoint.h"
#include "Model/WordEmbeddingsModel.h"

#include "Datapoint/MatrixInverseDatapoint.h"
#include "Model/MatrixInverseModel.h"

#include "Datapoint/LSDatapoint.h"
#include "Model/LSModel.h"

#endif
