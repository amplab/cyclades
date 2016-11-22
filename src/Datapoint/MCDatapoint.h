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

#ifndef _MCDATAPOINT_
#define _MCDATAPOINT_

#include <tuple>
#include <sstream>
#include "Datapoint.h"

class MCDatapoint : public Datapoint {
 private:
    double label;
    std::vector<double> weights;
    std::vector<int> coordinates;

    void Initialize(const std::string &input_line) {
	// Allocate data for coordiantes / weights.
	coordinates.resize(2);
	weights.resize(2);

	// Expected input_line format: user_coord, movie_coord, rating.
	std::stringstream input(input_line);
	input >> coordinates[0] >> coordinates[1] >> label;
	weights[0] = weights[1] = label;
    }

 public:

    MCDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	Initialize(input_line);
    }
    ~MCDatapoint() {}

    void OffsetMovieCoord(int offset) {
	coordinates[1] += offset;
    }

    std::vector<double> & GetWeights() override {
	return weights;
    }

    std::vector<int> & GetCoordinates() override {
	return coordinates;
    }

    int GetNumCoordinateTouches() override {
	return 2;
    }
};

#endif
