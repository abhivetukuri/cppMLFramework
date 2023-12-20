#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>

class Perceptron {
private:
    std::vector<double> weights;

public:
    Perceptron();
    void train(const std::vector<std::vector<double>>& trainSet, double lr, int numOfEpochs);
    double predict(const std::vector<double>& data) const;
};

#endif // PERCEPTRON_H
