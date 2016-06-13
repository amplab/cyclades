#ifndef _DATASET_READER_
#define _DATASET_READER_

#include <fstream>
#include "Model/Model.h"
#include "Datapoint/Datapoint.h"

class DatasetReader {
 public:
    template<class DATAPOINT_CLASS>
    static void ReadDataset(std::string &input_file,
			    std::vector<DATAPOINT_CLASS> &datapoints,
			    Model &model) {
	std::ifstream data_file_input(input_file);

	if (!data_file_input) {
	    std::cerr << "DatasetReader: Could not open file - " << input_file << std::endl;
	    exit(0);
	}

	// 1st line : model input line.
	std::string first_line;
	std::getline(data_file_input, first_line);
	model.Initialize(first_line);

	// 2nd line : number of datapoints.
	std::string num_datapoints_string;
	int num_datapoints;
	std::getline(data_file_input, num_datapoints_string);
	num_datapoints = stoi(num_datapoints_string);
	datapoints.resize(num_datapoints);

	// 3rd line+ : datapoint initialization.
	std::string datapoint_line;
	int datapoint_count = 0;
	while (std::getline(data_file_input, datapoint_line)) {
	    if (datapoint_count >= num_datapoints) {
		datapoint_count++;
		break;
	    }
	    datapoints[datapoint_count++].Initialize(datapoint_line);
	}

	// Some error checking.
	if (datapoint_count != num_datapoints) {
	    std::cerr << "DatasetReader: datapoint_count != num_datapoints" << std::endl;
	    exit(0);
	}
    }
};

#endif
