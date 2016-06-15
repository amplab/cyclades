#ifndef _DATASET_READER_
#define _DATASET_READER_

#include <fstream>
#include "Model/Model.h"
#include "Datapoint/Datapoint.h"

class DatasetReader {
 public:
    template<class DATAPOINT_CLASS, class MODEL_CLASS>
    static void ReadDataset(std::string &input_file,
			    std::vector<Datapoint *> &datapoints,
			    Model *&model) {
	// Allocate model.
	std::ifstream data_file_input(input_file);

	if (!data_file_input) {
	    std::cerr << "DatasetReader: Could not open file - " << input_file << std::endl;
	    exit(0);
	}

	// 1st line : model input line.
	std::string first_line;
	std::getline(data_file_input, first_line);
	model = new MODEL_CLASS(first_line);

	if (datapoints.size() != 0) {
	    std::cerr << "DatasetReader: datapoints is not empty." << std::endl;
	    exit(0);
	}

	// 2nd line+ : datapoint initialization.
	std::string datapoint_line;
	int datapoint_count = 0;
	while (std::getline(data_file_input, datapoint_line)) {
	    datapoints.push_back(new DATAPOINT_CLASS(datapoint_line));
	}
    }
};

#endif
