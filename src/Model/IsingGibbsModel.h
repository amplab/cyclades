#ifndef _ISINGGIBBSMODEL_
#define _ISINGGIBBSMODEL_

#include "Model.h"

DEFINE_double(gibbs_beta, 1, "Inverse temperature for Ising gibbs model");

class IsingGibbsModel : public Model {
 private:
    int n_points;
    bool is_2d_lattice;
    int *model;

    void Initialize(const std::string &input_line) {
	// Expect a single number representing # of points and whether it's a 2d lattice;
	std::stringstream input(input_line);
	input >> n_points >> is_2d_lattice;

	// Create and initialize random ising gibbs model state.
	model = (int *)malloc(sizeof(int) * n_points);
	for (int i = 0; i < n_points; i++) {
	    if (rand() % 2 == 0)  {
		model[i] = -1;
	    }
	    else {
		model[i] = 1;
	    }
	}
    }

    void Print2DState() {
	std::string state_string = "";
	int length = sqrt(n_points);
	for (int i = 0; i < length; i++) {
	    for (int j = 0; j < length; j++) {
		if (model[i*length+j] == 1) {
		    state_string += "1";
		}
		else if (model[i*length+j] == -1) {
		    state_string += "0";
		}
		else {
		    std::cerr << "IsingGibbsModel: Something went wrong..." << std::endl;
		    exit(0);
		}
	    }
	    state_string += "\n";
	}
	system("clear");
	std::cout << state_string << std::endl;
    }

 public:
    IsingGibbsModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~IsingGibbsModel() {
	delete [] model;
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	if (is_2d_lattice) {
	    Print2DState();
	}
	return 0;
    }

    void ComputeGradient(Datapoint *datapoint, Gradient *gradient, int thread_num) override {
	// There really is no gradient for the ising model. So just keep reference of the datapoint.
	GibbsGradient *grd = (GibbsGradient *)gradient;
	grd->datapoint = datapoint;
    }

    void ApplyGradient(Gradient *gradient) override {
	GibbsGradient *grd = (GibbsGradient *)gradient;
	GibbsDatapoint *datapoint = (GibbsDatapoint *)grd->datapoint;
	int index = datapoint->coord;
	int product_with_1 = 0;
	int product_with_neg_1 = 0;
	for (const auto &neighbor_index : datapoint->GetCoordinates()) {
	    product_with_1 += model[neighbor_index];
	    product_with_neg_1 += model[neighbor_index] * -1;
	}

	double p1 = exp(FLAGS_gibbs_beta * (double)product_with_1);
	double p2 = exp(FLAGS_gibbs_beta * (double)product_with_neg_1);
	double prob_1 = p1 / (p1+p2);
	double selection = ((double)rand() / (RAND_MAX));
	if (selection <= prob_1) {
	    model[index] = 1;
	}
	else {
	    model[index] = -1;
	}
    }

    int NumParameters() override {
	return n_points;
    }
};

#endif
