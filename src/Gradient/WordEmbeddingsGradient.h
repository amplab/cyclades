#ifndef _WORDEMBEDDINGSGRADIENT_
#define _WORDEMBEDDINGSGRADIENT_

#include "Gradient.h"

class WordEmbeddingsGradient : public Gradient {
 public:
    double gradient_coefficient;
    Datapoint *datapoint;

    void Clear() override {
	datapoint = NULL;
	gradient_coefficient = 0;
    }

    void SetUp(Model *model) override {

    }

    WordEmbeddingsGradient() {}

    ~WordEmbeddingsGradient() {}
};

#endif
