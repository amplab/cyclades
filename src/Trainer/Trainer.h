#ifndef _TRAINER_
#define _TRAINER_

#include <limits.h>
#include <float.h>

DEFINE_bool(random_batch_processing, false, "Process batches in random order. Note this may disrupt catch-up.");
DEFINE_bool(random_per_batch_datapoint_processing, false, "Process datapoints in random order per batch. Note this may disrupt catch-up.");
DEFINE_int32(interval_print, 1, "Interval in which to print the loss.");

// Contains times / losses / etc
struct TrainStatistics {
    std::vector<double> times;
    std::vector<double> losses;
};

typedef struct TrainStatistics TrainStatistics;

template<class GRADIENT_CLASS>
class Trainer {
protected:

    void TrackTimeLoss(double cur_time, double cur_loss, TrainStatistics *stats) {
	stats->times.push_back(cur_time);
	stats->losses.push_back(cur_loss);
    }

    void PrintPartitionTime(Timer &timer) {
	printf("Partition Time(s): %f\n", timer.Elapsed());
    }

    void PrintTimeLoss(double cur_time, double cur_loss) {
	printf("Time(s): %f\tLoss: %lf\n", cur_time, cur_loss);
    }

    void EpochBegin(int epoch, Timer &gradient_timer, Model *model, const std::vector<Datapoint *> &datapoints, TrainStatistics *stats) {
	double cur_time = gradient_timer.Elapsed();
	double cur_loss = model->ComputeLoss(datapoints);
	this->TrackTimeLoss(cur_time, cur_loss, stats);
	if (FLAGS_print_loss_per_epoch && epoch % FLAGS_interval_print == 0) {
	    this->PrintTimeLoss(cur_time, cur_loss);
	}
    }

public:
    Trainer() {
	// Some error checking.
	if (FLAGS_n_threads > std::thread::hardware_concurrency()) {
	    std::cerr << "Trainer: Number of threads is greater than the number of physical cores." << std::endl;
	    //exit(0);
	}

	// Basic set up, like pinning to core, setting number of threads.
	omp_set_num_threads(FLAGS_n_threads);
#pragma omp parallel
	{
	    pin_to_core(omp_get_thread_num());
	}
    }
    virtual ~Trainer() {}

    // Main training method.
    virtual TrainStatistics Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) = 0;
};

#endif
