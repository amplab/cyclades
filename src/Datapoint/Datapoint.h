#ifndef _DATAPOINT_
#define _DATAPOINT_

class Datapoint {
 public:
    Datapoint() {}
    virtual ~Datapoint() {}

    // Initialize given datapoint given input data line.
    virtual void Initialize(const std::string &input_line) = 0;

    // Write label of the given datapoint to pointer.
    // Assume memory is allocated.
    virtual double GetLabel() = 0;

    // Write data of datapoint to pointer.
    // Assume memory is allocated.
    virtual const std::vector<double> & GetData() = 0;

    // Get number of coordinates accessed by the datapoint.
    virtual int GetNumCoordinateTouches() = 0;
};

#endif
