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
	updater = new SGDUpdater<GRADIENT_CLASS>(FLAGS_n_threads);
    }
    if (!updater) {
	std::cerr << "Main: updater class not chosen." << std::endl;
	exit(0);
    }

    // Create trainer depending on flag.
    Trainer<GRADIENT_CLASS> *trainer;
    if (FLAGS_cyclades) {
	trainer = new CycladesTrainer<GRADIENT_CLASS>();
    }
    else {
	trainer = new HogwildTrainer<GRADIENT_CLASS>();
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
    else if (FLAGS_least_squares) {
	Run<LSModel, LSDatapoint, LSGradient>();
    }
}
