#ifndef _DEFINES_
#define _DEFINES_

#include <gflags/gflags.h>

DEFINE_int32(n_epochs, 100, "Number of passes of data in training.");
DEFINE_int32(n_threads, 2, "Number of threads in parallel during training.");
DEFINE_double(learning_rate, .001, "Learning rate.");
DEFINE_bool(cyclades, true, "Cyclades training if true, Hogwild training if false.");

#endif
