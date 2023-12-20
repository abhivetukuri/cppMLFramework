#include "Perceptron.h"
#include "LinearRegression.h"
#include <iostream>
#include <vector>

int main() {
    // Example usage of Perceptron
    Perceptron perceptron;
    std::vector<std::vector<double>> perceptronTrainSet = { /* your training data */ };
    perceptron.train(perceptronTrainSet, 0.1, 10);
    
    // Example usage of Linear Regression
    LinearRegression linearReg;
    std::vector<std::vector<double>> linearRegTrainSet = { /* your training data */ };
    linearReg.train(linearRegTrainSet, 0.01, 100);
    
    return 0;
}
