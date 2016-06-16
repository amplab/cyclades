#ifndef _MCMODEL_
#define _MCMODEL_

#include <sstream>
#include "Model.h"

DEFINE_int32(rlength, 100, "Length of vector in matrix completion.");

class MCModel : public Model {
 private:
    double *model;
    int n_users;
    int n_movies;
    int rlength;

    void InitializePrivateModel() {
	for (int i = 0; i < n_users+n_movies; i++) {
	    for (int j = 0; j < rlength; j++) {
		model[i*rlength+j] = ((double)rand()/(double)RAND_MAX);
	    }
	}
    }

    void Initialize(const std::string &input_line) {
	// Expected input_line format: N_USERS N_MOVIES.
	std::stringstream input(input_line);
	input >> n_users >> n_movies;
	rlength = FLAGS_rlength;

	// Allocate memory.
	model = new double[(n_users+n_movies) * rlength];
	if (!model) {
	    std::cerr << "MCModel: Error allocating model" << std::endl;
	    exit(0);
	}

	// Initialize private model.
	InitializePrivateModel();
    }

 public:
    MCModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~MCModel() {
	delete model;
    }

    void SetUp(const std::vector<Datapoint *> &datapoints) override {
	// Update the movies coordinates to reference the second
	// chunk of the model. Do this by offsetting the coordinates
	// by n_users.
	for (const auto & datapoint : datapoints) {
	    ((MCDatapoint *)datapoint)->OffsetMovieCoord(n_users);
	}
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
	for (int i = 0; i < datapoints.size(); i++) {
	    Datapoint *datapoint = datapoints[i];
	    const std::vector<double> & labels = datapoint->GetLabels();
	    const std::vector<int> & coordinates = datapoint->GetCoordinates();
	    double label = labels[0];
	    int x = coordinates[0];
	    int y = coordinates[1];
	    double cross_product = 0;
	    for (int j = 0; j < rlength; j++) {
		cross_product += model[x*rlength+j] * model[y*rlength+j];
	    }
	    double difference = cross_product - label;
	    loss += difference * difference;
	}
	return loss / datapoints.size();
    }
};

#endif
