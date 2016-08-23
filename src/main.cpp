// Sample call: ./cyclades -matrix_inverse -n_threads=2  -cyclades_trainer  -cyclades_batch_size=500  -learning_rate=.000001 --print_partition_time -n_epochs=20 -sgd -print_loss_per_epoch --data_file="data/nh2010/nh2010.data"

#include <iostream>
#include "defines.h"

template<class MODEL_CLASS, class DATAPOINT_CLASS, class GRADIENT_CLASS>
void Run() {

    // Initialize model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MODEL_CLASS, DATAPOINT_CLASS>(FLAGS_data_file, datapoints, model);
    model->SetUp(datapoints);

    // Create updater.
    Updater<GRADIENT_CLASS> *updater = NULL;
    if (FLAGS_sgd) {
	updater = new SGDUpdater<GRADIENT_CLASS>(model, datapoints, FLAGS_n_threads);
    }
    else if (FLAGS_minibatch_sgd) {
	updater = new MinibatchSGDUpdater<GRADIENT_CLASS>(model, datapoints, FLAGS_n_threads);
    }
    if (!updater) {
	std::cerr << "Main: updater class not chosen." << std::endl;
	exit(0);
    }

    // Create trainer depending on flag.
    Trainer<GRADIENT_CLASS> *trainer = NULL;
    if (FLAGS_cache_efficient_hogwild_trainer) {
	trainer = new CacheEfficientHogwildTrainer<GRADIENT_CLASS>();
    }
    if (FLAGS_cyclades_trainer) {
	trainer = new CycladesTrainer<GRADIENT_CLASS>();
    }
    else if (FLAGS_hogwild_trainer) {
	trainer = new HogwildTrainer<GRADIENT_CLASS>();
    }
    else if (FLAGS_minibatch_trainer) {
	trainer = new MinibatchTrainer<GRADIENT_CLASS>();
    }
    if (!trainer) {
	std::cerr << "Main: training method not chosen." << std::endl;
	exit(0);
    }
    trainer->Train(model, datapoints, updater);

    // Delete trainer.
    delete trainer;

    // Delete model and datapoints.
    delete model;
    for_each(datapoints.begin(), datapoints.end(), std::default_delete<Datapoint>());

    // Delete updater.
    delete updater;
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_matrix_completion) {
	Run<MCModel, MCDatapoint, MCGradient>();
    }
    else if (FLAGS_dense_least_squares) {
	Run<DenseLSModel, DenseLSDatapoint, DenseLSGradient>();
    }
    else if (FLAGS_word_embeddings) {
	Run<WordEmbeddingsModel, WordEmbeddingsDatapoint, WordEmbeddingsGradient>();
    }
    else if (FLAGS_matrix_inverse) {
	Run<MatrixInverseModel, MatrixInverseDatapoint, MatrixInverseGradient>();
    }
}
