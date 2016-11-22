// Make: make simple_ls
// Sample run: ./simple_ls --sparse_sgd --n_epochs=10  --learning_rate=1e-1 --print_loss_per_epoch --hogwild_trainer --n_threads=1 --data_file='./examples/simple_ls_data'

#include <iostream>
#include <vector>
#include "../src/run.h"
#include "../src/defines.h"

/* Minimize the equation sum (a_i x - b_i)^2
 * Each row of the A matrix (a_i) is represented by a single data point.
 * Also store b_i for each corresponding a_i.
 */
class SimpleLSDatapoint : public Datapoint {
public:
    std::vector<double> weights;
    std::vector<int> coordinates;
    double label;

    SimpleLSDatapoint(const std::string &input_line, int order) : Datapoint(input_line, order) {
	// Create a string stream from the input line.
	// This lets us read from the line as if reading via cin.
	std::stringstream in(input_line);

	// We expect a sparse row from the A matrix, a_i, as well as the
	// corresponding value of b_i. The first value of the line is the number
	// of nnz values in the row of the matrix.
	// E.g:
	// n index_1 value_1 index_2 value_2 .... index_n value_n label
	int n;
	in >> n;
	weights.resize(n);
	coordinates.resize(n);
	for (int i = 0; i < n; i++) {
	    in >> coordinates[i];
	    in >> weights[i];
	}
	in >> label;
    }

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

/* Minimize the equation sum (a_i x - b_i)^2
 * Represents the x model.
 *
 * The gradient at a datapoint a_i is
 * d/dx f(x) = 2(a_i * x - b_i) x.
 * Therefore, in terms of [∇f(x)]_j = λ_j * x_j − κ_j + h_bar_j(x),
 * which defines the gradient at a datapoint at a coordinate j of the model,
 * we have λ_j = 0, x_j = 0, h_bar_j(x) = 2(a_i * x - b_i) a_i.
 *
 */
class SimpleLSModel : public Model {
private:
    double dot(SimpleLSDatapoint *a_i, std::vector<double> &x) {
	double product = 0;
	for (int i = 0; i < a_i->GetNumCoordinateTouches(); i++) {
	    int index = a_i->GetCoordinates()[i];
	    double value = a_i->GetWeights()[i];
	    product += value * x[index];
	}
	return product;
    }

public:
    std::vector<double> x;

    SimpleLSModel(const std::string &input_line) {
	// Create a string stream from the input line.
	// This lets us read from the line as if reading via cin.
	std::stringstream in(input_line);

	// We expect a single integer describing the number
	// of coordinates of the model.
	int num_coordinates;
	in >> num_coordinates;

	// Preallocate the x model and randomly initialize.
	x.resize(num_coordinates);
	for (int i = 0; i < x.size(); i++) {
	    x[i] = ((double)rand()/(double)RAND_MAX);
	}
    }

    // Minimize loss = sum (a_i * x - b)^2.
    double ComputeLoss(const std::vector<Datapoint *> &datapoints) override {
	double loss = 0;

	// Note that ComputeLoss is called by a SINGLE thread.
	// It is possible to parallelize this via
	// #pragma omp parallel for
	for (int i = 0; i < datapoints.size(); i++) {
	    SimpleLSDatapoint *a_i = (SimpleLSDatapoint *)datapoints[i];
	    double b_i = a_i->label;
	    double dot_product = dot(a_i, x);
	    loss += (dot_product - b_i) * (dot_product - b_i);
	}

	std::cout << "Model Parameters: " << std::endl;
	for (int i = 0; i < NumParameters(); i++) {
	    std::cout << x[i] << " ";
	}
	std::cout << std::endl;

	return loss / datapoints.size();
    }

    int NumParameters() override {
	return x.size();
    }

    int CoordinateSize() override {
	// Each coordinate in the model is a single scalar.
	return 1;
    }

    std::vector<double> &ModelData() override {
	return x;
    }

    void Lambda(int coordinate, double &out, std::vector<double> &local_model) override {
	out = 0;
    }

    void Kappa(int coordinate, std::vector<double> &out, std::vector<double> &local_model) override {
	out[0] = 0;
    }

    // h_bar_j(x) = 2(a_i * x - b_i) a_i
    // We can just precompute the each h_bar_j directly.
    void PrecomputeCoefficients(Datapoint *datapoint, Gradient *g, std::vector<double> &local_model) override {
	// We need to make sure g->coeffs can store the gradient to the model.
	if (g->coeffs.size() != 1) g->coeffs.resize(NumParameters());

	// Compute 2(a_i * x - b_i).
	SimpleLSDatapoint *a_i = (SimpleLSDatapoint *)datapoint;
	double b_i = a_i->label;
	double coefficient = 2 * (dot(a_i, local_model) - b_i);

	// For each nnz weight of the data point, set g->coeffs appropriately.
	for (int i = 0; i < datapoint->GetNumCoordinateTouches(); i++) {
	    int index = datapoint->GetCoordinates()[i];
	    double weight = datapoint->GetWeights()[i];
	    g->coeffs[index] = coefficient * weight;
	}
    }

    // Since g->coeffs[0] = 2(a_i * x - b_i),
    // The gradient is g->coeffs[0] * a_ij (i'th datapoint, j'th
    void H_bar(int coordinate, std::vector<double> &out, Gradient *g, std::vector<double> &local_model) override {
	out[0] = g->coeffs[coordinate];
    }
};

int main(int argc, char **argv) {
    std::cout << "Simple least squares custom optimization example:" << std::endl;
    std::cout << "Minimize the equation ||Ax-b||^2 = Minimize Sum (a_i x - b_i)^2" << std::endl;

    // Initialize gflags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Run<SimpleLSModel, SimpleLSDatapoint>();
}
