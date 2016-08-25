#ifndef _ISINGGIBBSMODEL_
#define _ISINGGIBBSMODEL_

#include "Model.h"

DEFINE_double(gibbs_beta, 1, "Inverse temperature for Ising gibbs model");

class IsingGibbsModel : public Model {
 private:
    int n_points;
    int *model;

    void Initialize(const std::string &input_line) {
	// Expect a single number representing # of points;
	std::stringstream input(input_line);
	input >> n_points;

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

 public:
    IsingGibbsModel(const std::string &input_line) {
	Initialize(input_line);
    }

    ~IsingGibbsModel() {
	delete [] model;
    }

    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
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
