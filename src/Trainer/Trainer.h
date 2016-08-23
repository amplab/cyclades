#ifndef _TRAINER_
#define _TRAINER_

template<class GRADIENT_CLASS>
class Trainer {
protected:
    void PrintPartitionTime(Timer &timer) {
	printf("Partition Time(s): %f\n", timer.Elapsed());
    }

    void PrintTimeLoss(Timer &timer, Model *model, const std::vector<Datapoint *> &datapoints) {
	printf("Time(s): %f\tLoss: %lf\n", timer.Elapsed(), model->ComputeLoss(datapoints));
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
    virtual void Train(Model *model, const std::vector<Datapoint *> & datapoints, Updater<GRADIENT_CLASS> *updater) = 0;
};

#endif
