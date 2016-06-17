#ifndef _DEFINES_
#define _DEFINES_

#include <omp.h>
#include <thread>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include "Datapoint/Datapoint.h"
#include "Datapoint/MCDatapoint.h"
#include "Model/Model.h"
#include "Model/MCModel.h"
#include "DatasetReader.h"
#include "Updater/Updater.h"
#include "Updater/SGDUpdater.h"
#include "DatapointPartitions/DatapointPartitions.h"
#include "Partitioner/Partitioner.h"
#include "Partitioner/BasicPartitioner.h"

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

#include "Trainer/Trainer.h"
#include "Trainer/CycladesTrainer.h"
#include "Trainer/HogwildTrainer.h"

#endif
