// To compare against file:
// ./cyclades --interval_print=20000 --print_loss_per_epoch --should_compare_to_compare_distribution_file=true  --should_write_to_distribution_output_file=false  --cyclades_batch_size=200  -ising_gibbs -n_threads=16 --random_per_batch_datapoint_processing  -hogwild_trainer  -print_partition_time -n_epochs=200000 -sgd --data_file="data/gibbs/gibbs.data"
//
// To generate the distribution file:
// ./cyclades --write_distribution_interval=200000 --should_compare_to_compare_distribution_file=false  --should_write_to_distribution_output_file  -ising_gibbs -n_threads=1 --random_per_batch_datapoint_processing  -hogwild_trainer  -print_partition_time -n_epochs=2000000 -sgd --data_file="data/gibbs/gibbs.data"

#ifndef _ISINGGIBBSMODEL_
#define _ISINGGIBBSMODEL_

#include "Model.h"
#include <random>
#include <thread>

DEFINE_double(gibbs_beta, .2, "Inverse temperature for Ising gibbs model.");
DEFINE_double(gibbs_prior_weight, 1, "Prior weights for Ising gibbs model.");
DEFINE_string(distribution_output_file, "IsingGibbsDistributionFile.out", "Output file to write distribution of states.");
DEFINE_int32(write_distribution_interval, 1000, "Interval to write distribution to distribution_output_file.");
DEFINE_bool(should_write_to_distribution_output_file, false, "Should write distribution to distribution_output_file.");
DEFINE_string(compare_distribution_file, "IsingGibbsDistributionFile.cmp", "Comparison file containing distribution to compare the current distribution to.");
DEFINE_bool(should_compare_to_compare_distribution_file, true, "Should compare to compare_distribution_file.");

struct VertexDistribution {
    int n_negatives, n_positives;
};

typedef struct VertexDistribution VertexDistribution;

class IsingGibbsModel : public Model {
 private:
    int n_points;
    bool is_2d_lattice;
    std::vector<VertexDistribution> states_distribution;
    std::vector<VertexDistribution> compare_distribution;
    int n_epochs_finished;
    int *model;

    void Initialize(const std::string &input_line) {
	n_epochs_finished = 0;

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

	// States distributions loading.
	for (int i = 0; i < n_points; i++) {
	    states_distribution.push_back({0,0});
	    compare_distribution.push_back({0,0});
	}
	if (FLAGS_should_compare_to_compare_distribution_file) {
	    LoadStatesDistributionFromFile(compare_distribution, FLAGS_compare_distribution_file);
	}
    }

    int ThreadsafeRand(const int & min, const int & max) {
	static thread_local std::mt19937_64 generator((omp_get_thread_num() + 1) * static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
	std::uniform_int_distribution<int> distribution(min,max);
	return distribution(generator);
    }

    double CompareStatesDistribution(std::vector<VertexDistribution> &d1,
				     std::vector<VertexDistribution> &d2) {
	double error = 0;
	#pragma omp parallel for num_threads(FLAGS_n_threads) reduction(+:error)
	for (int i = 0; i < n_points; i++) {
	    double p1 = d1[i].n_positives / (double)(d1[i].n_positives + d1[i].n_negatives);
	    double p2 = d2[i].n_positives / (double)(d2[i].n_positives + d2[i].n_negatives);
	    error += pow(p1-p2, 2);
	}
	return sqrt(error);
    }

    void LoadStatesDistributionFromFile(std::vector<VertexDistribution> &d, std::string filename) {
	std::ifstream f_in(filename);
	int n_points_in_file;
	f_in >> n_points_in_file;
	if (n_points_in_file != n_points) {
	    std::cout << "IsingGibbsModel: n_points in comparison file does not match n_points in model" << std::endl;
	    exit(0);
	}
	for (int i = 0; i < n_points_in_file; i++) {
	    f_in >> d[i].n_positives;
	    f_in >> d[i].n_negatives;
	}
	f_in.close();
    }

    void WriteStatesDistributionToFile(std::vector<VertexDistribution> &d, std::string outfile) {
	std::ofstream f_out(outfile);
	f_out << d.size() << std::endl;
	for (int i = 0; i < d.size(); i++) {
	    f_out << d[i].n_positives << " " << d[i].n_negatives << std::endl;
	}
	f_out.close();
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
	if (FLAGS_should_compare_to_compare_distribution_file) {
	    return CompareStatesDistribution(states_distribution, compare_distribution);
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
	int tendency = datapoint->tendency;
	int product_with_1 = 0;
	int product_with_neg_1 = 0;
	for (const auto &neighbor_index : datapoint->GetCoordinates()) {
	    product_with_1 += model[neighbor_index];
	    product_with_neg_1 += model[neighbor_index] * -1;
	}

	double negative_tendency = -1 * tendency;
	double positive_tendency = tendency;

	double p1 = exp(FLAGS_gibbs_beta * (double)product_with_1 + FLAGS_gibbs_prior_weight * positive_tendency);
	double p2 = exp(FLAGS_gibbs_beta * (double)product_with_neg_1 + FLAGS_gibbs_prior_weight * negative_tendency);
	double prob_1 = p1 / (p1+p2);
	double selection = ThreadsafeRand(0, RAND_MAX) / (double)RAND_MAX;
	if (selection <= prob_1) {
	    model[index] = 1;
	}
	else {
	    model[index] = -1;
	}
    }

    void EpochFinish() override {

	// Update distribution of states.
	for (int i = 0; i < n_points; i++) {
	    if (model[i] < 0) {
		states_distribution[i].n_negatives++;
	    }
	    else {
		states_distribution[i].n_positives++;
	    }
	}
	n_epochs_finished++;

	// Write distribution of states to file.
	if (n_epochs_finished % FLAGS_write_distribution_interval == 0 &&
	    FLAGS_should_write_to_distribution_output_file) {
	    std::cout << "IsingGibbsModel: Writing distribution to " <<
		FLAGS_distribution_output_file.c_str() << "." << std::endl;
	    WriteStatesDistributionToFile(states_distribution, FLAGS_distribution_output_file);
	}
    }

    int NumParameters() override {
	return n_points;
    }
};

#endif
