#include <iostream>
#include "defines.h"

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // For Debug for now.
    Model model;
    std::vector<Datapoint> datapoints;
    DatasetReader::ReadDataset(FLAGS_data_file, datapoints, model);
}
