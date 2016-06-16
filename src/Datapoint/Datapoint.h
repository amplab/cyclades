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

    // Get labels corresponding to the corresponding coordinates of GetCoordinates().
    virtual const std::vector<double> & GetLabels() = 0;

    // Get coordinates corresponding to labels of GetLabels().
    virtual const std::vector<int> & GetCoordinates() = 0;

    // Get number of coordinates accessed by the datapoint.
    virtual int GetNumCoordinateTouches() = 0;

    // Get the order of a datapoint (equivalent to id).
    virtual int GetOrder() {
	return order;
    }
};

#endif
