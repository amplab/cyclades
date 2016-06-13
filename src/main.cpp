#include <iostream>
#include "defines.h"

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // For Debug for now.
    MCModel model;
    std::vector<MCDatapoint> datapoints;
    DatasetReader::ReadDataset<MCDatapoint>(FLAGS_data_file, datapoints, model);
}
