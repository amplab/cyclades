#ifndef _DATAPOINT_
#define _DATAPOINT_

class Datapoint {
 private:
    int order;
 public:
    Datapoint() {}
    Datapoint(const std::string &input_line, int order) {
	this->order = order;
    }
    virtual ~Datapoint() {}

    // Write label of the given datapoint to pointer.
    // Assume memory is allocated.
    virtual double GetLabel() = 0;

    // Write data of datapoint to pointer.
    // Assume memory is allocated.
    virtual void * GetData() = 0;

    // Get number of coordinates accessed by the datapoint.
    virtual int GetNumCoordinateTouches() = 0;

    // Get the order of a datapoint (equivalent to id).
    virtual int GetOrder() {
	return order;
    }
};

#endif
