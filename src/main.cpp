#include <iostream>
#include "defines.h"

void Run() {
    // Initialize model and datapoints.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MCModel, MCDatapoint>(FLAGS_data_file, datapoints, model);

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
    Run();
}
