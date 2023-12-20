#include "Perceptron.h"
#include <cmath>
#include <iostream>

Perceptron::Perceptron() {}

void Perceptron::train(const std::vector<std::vector<double>>& trainSet, double lr, int numOfEpochs) {
    int numOfWeights = 1 + (trainSet[0].size() - 1);
    weights.resize(numOfWeights, 0);

    for (int e = 0; e < numOfEpochs; e++) {
        double sumOfError = 0;
        for (const auto& data : trainSet) {
            double label = data.back();
            double pred = predict(data);
            double error = label - pred;
            sumOfError += error * error;

            weights[0] = weights[0] + lr * error;
            for (int i = 1; i < weights.size(); i++) {
                weights[i] = weights[i] + lr * error * data[i - 1];
            }
        }
        std::cout << "Epoch=" << e << ", Sum of Error=" << sumOfError << std::endl;
    }
}

double Perceptron::predict(const std::vector<double>& data) const {
    double z = weights[0];
    for (size_t i = 1; i < weights.size(); i++) {
        z += weights[i] * data[i - 1];
    }
    return (z >= 0.0) ? 1.0 : 0.0;
}
