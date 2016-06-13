#ifndef _MCMODEL_
#define _MCMODEL_

#include <sstream>

DEFINE_int32(rlength, 100, "Length of vector in matrix completion.");

class MCModel : public Model {
 private:
    double *v_model;
    double *u_model;
 public:
    MCModel() {}
    ~MCModel() {}
    void Initialize(std::string &input_line) override {
	// Expected input_line format: N_USERS N_MOVIES.
	std::stringstream input(input_line);
	int n_users, n_movies;
	input >> n_users >> n_movies;
	int rlength = FLAGS_rlength;
	v_model = (double *)malloc(sizeof(double) * n_users * rlength);
	u_model = (double *)malloc(sizeof(double) * n_movies * rlength);
	if (!v_model || !u_model) {
	    std::cerr << "MCModel: Error allocating model" << std::endl;
	    exit(0);
	}
    }
};

#endif
