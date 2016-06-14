#include <iostream>
#include "defines.h"

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // For Debug for now.
    Model *model;
    std::vector<Datapoint *> datapoints;
    DatasetReader::ReadDataset<MCDatapoint, MCModel>(FLAGS_data_file, datapoints, &model);

    delete model;
    for_each(datapoints.begin(), datapoints.end(), std::default_delete<Datapoint>());
}
