#include "Perceptron.h"
#include "LinearRegression.h"
#include <iostream>
#include <vector>

int main()
{
    std::cout << "Perceptron Model Example:" << std::endl;

    std::vector<std::vector<double>> perceptronData = {
        {1.0, 2.0, 0.0},
        {1.5, 1.5, 0.0},
        {5.0, 2.0, 1.0},
    };

    Perceptron perceptron;
    perceptron.train(perceptronData, 0.1, 10);

    std::vector<double> perceptronTestInput = {2.0, 3.0};
    double perceptronPrediction = perceptron.predict(perceptronTestInput);
    std::cout << "Perceptron prediction for input {2.0, 3.0}: " << perceptronPrediction << std::endl;

    std::cout << "\nLinear Regression Model Example:" << std::endl;

    std::vector<std::vector<double>> linearRegData = {
        {1.0, 2.0},
        {2.0, 3.0},
        {4.0, 5.0},
    };

    LinearRegression linearReg;
    linearReg.train(linearRegData, 0.01, 100);

    double linearRegTestInput = 3.0;
    double linearRegPrediction = linearReg.predict(linearRegTestInput);
    std::cout << "Linear Regression prediction for input 3.0: " << linearRegPrediction << std::endl;

    return 0;
}
