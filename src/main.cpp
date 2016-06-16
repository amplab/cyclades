#include <iostream>
#include "defines.h"

template<class MODEL_CLASS, class DATAPOINT_CLASS>
void Run() {
    // Initialize model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MODEL_CLASS, DATAPOINT_CLASS>(FLAGS_data_file, datapoints, model);

    // Create trainer depending on flag.
    Trainer *trainer;
    if (FLAGS_cyclades) {
	trainer = new CycladesTrainer();
    }
    else {
	trainer = new HogwildTrainer();
    }
    trainer->Run(model, datapoints);

    // Delete trainer.
    delete trainer;

    // Delete model and datapoints.
    delete model;
    for_each(datapoints.begin(), datapoints.end(), std::default_delete<Datapoint>());
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    Run<MCModel, MCDatapoint>();
}
