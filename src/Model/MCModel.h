#ifndef _MCMODEL_
#define _MCMODEL_

#include <sstream>

DEFINE_int32(rlength, 100, "Length of vector in matrix completion.");

class MCModel : public Model {
 private:
    double *v_model;
    double *u_model;
    int n_users;
    int n_movies;
    int rlength;

    void InitializePrivateModels() {
	for (int i = 0; i < n_users; i++) {
	    for (int j = 0; j < rlength; j++) {
		v_model[i*rlength+j] = ((double)rand()/(double)RAND_MAX);
	    }
	}
	for (int i = 0; i < n_movies; i++) {
	    for (int j = 0; j < FLAGS_rlength; j++) {
		u_model[i*rlength+j] = ((double)rand()/(double)RAND_MAX);
	    }
	}
    }

    void Initialize(const std::string &input_line) {
	// Expected input_line format: N_USERS N_MOVIES.
	std::stringstream input(input_line);
	input >> n_users >> n_movies;
	rlength = FLAGS_rlength;

	// Allocate memory.
	v_model = new double[n_users * rlength];
	u_model = new double[n_movies * rlength];
	if (!v_model || !u_model) {
	    std::cerr << "MCModel: Error allocating model" << std::endl;
	    exit(0);
	}

	// Initialize private model.
	InitializePrivateModels();
    }

 public:
    MCModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~MCModel() {
	delete u_model;
	delete v_model;
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;
	for (int i = 0; i < datapoints.size(); i++) {
	    Datapoint *datapoint = datapoints[i];
	    double label = datapoint->GetLabel();
	    int *data = (int *)datapoint->GetData();
	    int x = data[0];
	    int y = data[1];
	    double cross_product = 0;
	    for (int j = 0; j < rlength; j++) {
		cross_product += v_model[x*rlength+j] * u_model[y*rlength+j];
	    }
	    double difference = cross_product - label;
	    loss += difference * difference;
	}
	return loss / datapoints.size();
    }
};

#endif
