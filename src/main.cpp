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

// Sample call: ./cyclades -matrix_inverse -n_threads=2  -cyclades_trainer  -cyclades_batch_size=500  -learning_rate=.000001 --print_partition_time -n_epochs=20 -sgd -print_loss_per_epoch --data_file="data/nh2010/nh2010.data"

#include <iostream>
#include "run.h"

// Flags for application types.
DEFINE_bool(matrix_completion, false, "Matrix completion application type.");
DEFINE_bool(fast_matrix_completion, false, "Matrix completion with custom sgd updater. Do not specify an extra updater (e.g: don't specify --sparse_sgd, etc)");
DEFINE_bool(word_embeddings, false, "W2V application type. Do NOT set an updater (E.G: sparse_sgd) if you want to use the default optimizer which optimizes C.");
DEFINE_bool(matrix_inverse, false, "Matrix inverse application type.");
DEFINE_bool(least_squares, false, "Sparse least squares application type.");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_matrix_completion) {
	Run<MCModel, MCDatapoint>();
    }
    else if (FLAGS_word_embeddings) {
	Run<WordEmbeddingsModel, WordEmbeddingsDatapoint, WordEmbeddingsSGDUpdater>();
    }
    else if (FLAGS_matrix_inverse) {
	Run<MatrixInverseModel, MatrixInverseDatapoint>();
    }
    else if (FLAGS_least_squares) {
	Run<LSModel, LSDatapoint>();
    }
    else if (FLAGS_fast_matrix_completion) {
	Run<MCModel, MCDatapoint, FastMCSGDUpdater>();
    }
}
