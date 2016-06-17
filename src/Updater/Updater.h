#ifndef _UPDATER_
#define _UPDATER_

template <class GRADIENT_CLASS>
class Updater {
 public:
    Updater(int n_threads) {}
    Updater() {}
    virtual ~Updater() {}

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint, int thread_num) = 0;
};

#endif
