#include "Perceptron.h"
#include "LinearRegression.h"
#include <iostream>
#include <vector>

int main() {
    // Example usage of Perceptron model
    std::cout << "Perceptron Model Example:" << std::endl;

    // Sample data for Perceptron
    // Format: {feature1, feature2, ..., label}
    std::vector<std::vector<double>> perceptronData = {
        {1.0, 2.0, 0.0},
        {1.5, 1.5, 0.0},
        {5.0, 2.0, 1.0},
        // ... add more data as needed
    };

    Perceptron perceptron;
    perceptron.train(perceptronData, 0.1, 10); // Train the model

    // Test the Perceptron model with a sample input
    std::vector<double> perceptronTestInput = {2.0, 3.0};
    double perceptronPrediction = perceptron.predict(perceptronTestInput);
    std::cout << "Perceptron prediction for input {2.0, 3.0}: " << perceptronPrediction << std::endl;

    // Example usage of Linear Regression model
    std::cout << "\nLinear Regression Model Example:" << std::endl;

    // Sample data for Linear Regression
    // Format: {feature, value}
    std::vector<std::vector<double>> linearRegData = {
        {1.0, 2.0},
        {2.0, 3.0},
        {4.0, 5.0},
        // ... add more data as needed
    };

    LinearRegression linearReg;
    linearReg.train(linearRegData, 0.01, 100); // Train the model

    // Test the Linear Regression model with a sample input
    double linearRegTestInput = 3.0;
    double linearRegPrediction = linearReg.predict(linearRegTestInput);
    std::cout << "Linear Regression prediction for input 3.0: " << linearRegPrediction << std::endl;

    return 0;
}
