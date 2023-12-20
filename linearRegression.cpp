#include "LinearRegression.h"
#include <cmath>
#include <iostream>

LinearRegression::LinearRegression() : intercept(0), slope(0) {}

void LinearRegression::train(const std::vector<std::vector<double>>& trainSet, double learningRate, int numOfEpochs) {
    for (int epoch = 0; epoch < numOfEpochs; ++epoch) {
        double sumError = 0;
        for (const auto& data : trainSet) {
            double x = data[0], y = data[1];
            double prediction = predict(x);
            double error = y - prediction;
            sumError += error * error;
            intercept += learningRate * error;
            slope += learningRate * error * x;
        }
        std::cout << "Epoch " << epoch << ", Error " << sumError << std::endl;
    }
}

double LinearRegression::predict(double x) const {
    return slope * x + intercept;
}
