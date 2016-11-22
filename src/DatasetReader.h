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

#ifndef _DATASET_READER_
#define _DATASET_READER_

#include <vector>
#include <fstream>
#include "Model/Model.h"
#include "Datapoint/Datapoint.h"

class DatasetReader {
 public:
    template<class MODEL_CLASS, class DATAPOINT_CLASS>
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
	    datapoints.push_back(new DATAPOINT_CLASS(datapoint_line, datapoint_count++));
	}
    }
};

#endif
