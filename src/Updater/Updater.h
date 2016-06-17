#ifndef _UPDATER_
#define _UPDATER_

class Updater {
 public:
    Updater() {}
    virtual ~Updater() {}

    // Main update method.
    virtual void Update(Model *model, Datapoint *datapoint) = 0;
};

#endif
