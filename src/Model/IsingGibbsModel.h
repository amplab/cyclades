#ifndef _ISINGGIBBSMODEL_
#define _ISINGGIBBSMODEL_

#include "Model.h"
#include <random>
#include <thread>

DEFINE_double(gibbs_beta, .2, "Inverse temperature for Ising gibbs model.");
DEFINE_string(distribution_output_file, "IsingGibbsDistributionFile.out", "Output file to write distribution of states.");
DEFINE_int32(write_distribution_interval, 1000, "Interval to write distribution to distribution_output_file.");
DEFINE_bool(should_write_to_distribution_output_file, false, "Should write distribution to distribution_output_file.");
DEFINE_string(compare_distribution_file, "IsingGibbsDistributionFile.cmp", "Comparison file containing distribution to compare the current distribution to.");
DEFINE_int32(compare_distribution_interval, 1000, "Interval to compare distribution from compare_distribution_file.");
DEFINE_bool(should_compare_to_compare_distribution_file, true, "Should compare to compare_distribution_file.");

class IsingGibbsModel : public Model {
 private:
    int n_points;
    bool is_2d_lattice;
    std::map<std::string, int> states_distribution;
    int n_states_accumulated;
    int *model;

    void Initialize(const std::string &input_line) {
	n_states_accumulated = 0;

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

    int ThreadsafeRand(const int & min, const int & max) {
	static thread_local std::mt19937_64 generator((omp_get_thread_num() + 1) * static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
	std::uniform_int_distribution<int> distribution(min,max);
	return distribution(generator);
    }

    double CompareStatesDistribution(std::map<std::string, int> &d1,
				     std::map<std::string, int> &d2) {
	double s1 = 0, s2 = 0;
	double error = 0;
	for (const auto &element : d1) s1 += element.second;
	for (const auto &element : d2) s2 += element.second;
	for (const auto &element : d1) {
	    double s1_prob = element.second / s1;
	    double s2_prob = 0;
	    if (d2.find(element.first) != d2.end()) {
		s2_prob = d2[element.first] / s2;
	    }
	    error += pow(s2_prob-s1_prob, 2);
	}
	return sqrt(error);
    }

    void LoadStatesDistributionFromFile(std::map<std::string, int> &d, std::string filename) {
	std::ifstream f_in(filename);
	while (f_in) {
	    std::string state;
	    int count;
	    f_in >> state;
	    if (!f_in) break;
	    f_in >> count;
	    d[state] = count;
	}
    }

    void WriteStatesDistributionToFile(std::map<std::string, int> &distr, std::string outfile) {
	// Doesn't seem like C++ has builtin tools to serialize a map.
	// Just store them in format : state count pairs.
	std::ofstream f_out(outfile);
	for (const auto &element : distr) {
	    f_out << element.first << " " << element.second << std::endl;
	}
	f_out.close();
    }

    std::string SerializeModel() {
	std::string state_string = "";
	for (int i = 0; i < n_points; i++) {
	    if (model[i] == 1) {
		state_string += "1";
	    }
	    else {
		state_string += "0";
	    }
	}
	return state_string;
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
	//double selection = ThreadsafeRand(0, RAND_MAX) / (double)RAND_MAX;
	double selection = .5;
	if (selection <= prob_1) {
	    model[index] = 1;
	}
	else {
	    model[index] = -1;
	}
    }

    void EpochFinish() override {

	// Update distribution of states.
	std::string model_string = SerializeModel();
	if (states_distribution.find(model_string) == states_distribution.end()) {
	    states_distribution[model_string] = 0;
	}
	states_distribution[model_string]++;
	n_states_accumulated++;

	// Write distribution of states to file.
	if (n_states_accumulated % FLAGS_write_distribution_interval == 0 &&
	    FLAGS_should_write_to_distribution_output_file) {
	    std::cout << "IsingGibbsModel: Writing distribution with " <<
		states_distribution.size() <<  " states to " <<
		FLAGS_distribution_output_file.c_str() << "." << std::endl;
	    WriteStatesDistributionToFile(states_distribution, FLAGS_distribution_output_file);
	}

	if (n_states_accumulated % FLAGS_compare_distribution_interval == 0 &&
	    FLAGS_should_compare_to_compare_distribution_file) {
	    std::map<std::string, int> compare_distr;
	    LoadStatesDistributionFromFile(compare_distr, FLAGS_compare_distribution_file);
	    std::cout << CompareStatesDistribution(states_distribution, compare_distr) << std::endl;
	}
    }

    int NumParameters() override {
	return n_points;
    }
};

#endif
