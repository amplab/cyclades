#ifndef _DEFINES_
#define _DEFINES_

#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <thread>
#include <map>
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
#include "Datapoint/MCDatapoint.h"
#include "Gradient/Gradient.h"
#include "Gradient/MCGradient.h"
#include "DatasetReader.h"
#include "Updater/Updater.h"
#include "Updater/SGDUpdater.h"
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
  //cpu_set_t cpuset;
  //CPU_ZERO(&cpuset);
  //CPU_SET(core, &cpuset);
  //pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

DEFINE_string(data_file, "blank", "Input data file.");
DEFINE_int32(n_epochs, 100, "Number of passes of data in training.");
DEFINE_int32(n_threads, 2, "Number of threads in parallel during training.");
DEFINE_double(learning_rate, .001, "Learning rate.");
DEFINE_bool(cyclades, true, "Cyclades training if true, Hogwild training if false.");
DEFINE_bool(print_loss_per_epoch, false, "Should compute and print loss every epoch.");
DEFINE_bool(print_partition_time, false, "Should print time taken to distribute datapoints across threads.");
DEFINE_bool(sgd, true, "Use the SGD update method.");
DEFINE_int32(batch_size, 5000, "Batch size for cyclades.");

#include "Partitioner/CycladesPartitioner.h"
#include "Model/MCModel.h"
#include "Trainer/Trainer.h"
#include "Trainer/CycladesTrainer.h"
#include "Trainer/HogwildTrainer.h"

#endif
