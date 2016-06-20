#ifndef _GRADIENT_
#define _GRADIENT_

class Gradient {
 public:
    Gradient() {}
    virtual ~Gradient() {}

    virtual void Add(const Gradient &other) {
	std::cerr << "Gradient: Add is not implemented." << std::endl;
    }
};

#endif
