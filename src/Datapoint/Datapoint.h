#ifndef _DATAPOINT_
#define _DATAPOINT_

class Datapoint {
 public:
    Datapoint() {}
    ~Datapoint() {}

    // Initialize given datapoint given input data line.
    virtual void Initialize(std::string &input_line) {
	// Override.
    }
};

#endif
