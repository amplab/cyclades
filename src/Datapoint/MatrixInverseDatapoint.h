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

#ifndef _MATRIXINVERSEDATAPOINT_
#define _MATRIXINVERSEDATAPOINT_

class MatrixInverseDatapoint : public Datapoint {
 private:
    int row;
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {

	std::stringstream input(input_line);

	// Expect format:
	// Row# index#1 weight1 index#2 weight2 ...
	input >> row;
	while (input) {
	    int index;
	    double weight;
	    input >> index;
	    if (!input) {
		break;
	    }
	    input >> weight;
	    coordinates.push_back(index);
	    weights.push_back(weight);
	    coordinate_weight_map[index] = weight;
	}
    }

 public:
    std::map<int, double> coordinate_weight_map;

    MatrixInverseDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~MatrixInverseDatapoint() {}

    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return coordinates.size();
    }
};

#endif
