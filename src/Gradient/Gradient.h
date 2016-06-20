#ifndef _GRADIENT_
#define _GRADIENT_

class Model;

class Gradient {
 public:
    Gradient() {}
    virtual ~Gradient() {}

    virtual void Clear() = 0;

    virtual void SetUp(Model *model) = 0;

    virtual void Add(const Gradient &other) {
	std::cerr << "Gradient: Add is not implemented." << std::endl;
    }
};

#endif
