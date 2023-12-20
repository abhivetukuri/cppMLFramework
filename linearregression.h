#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

class LinearRegression {
private:
    double intercept, slope;

public:
    LinearRegression();
    void train(const std::vector<std::vector<double>>& trainSet, double learningRate, int numOfEpochs);
    double predict(double x) const;
};

#endif // LINEAR_REGRESSION_H
