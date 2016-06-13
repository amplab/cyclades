#ifndef _MODEL_
#define _MODEL_

class Model {
 public:
    Model() {}
    ~Model() {}
    // Initialize model given input line from data file.
    virtual void Initialize(std::string &input_line) {
	// Override.
    }
};

#endif
