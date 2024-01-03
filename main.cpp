#include "Perceptron.h"
#include "LinearRegression.h"
#include "NaiveBayes.h"
#include "knn.h"
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

    // Example usage of Naive Bayes model
    std::cout << "\nNaive Bayes Model Example:" << std::endl;

    // Sample data for Naive Bayes
    // Format: {feature1, feature2, ..., label}
    std::vector<std::vector<double>> naiveBayesData = {
        {1.0, 20.0, 0},
        {2.0, 22.0, 0},
        {3.0, 18.0, 1},
        {2.0, 21.0, 1},
        // ... add more data as needed
    };

    NaiveBayes naiveBayes;
    naiveBayes.train(naiveBayesData); // Train the model

    // Test the Naive Bayes model with a sample input
    std::vector<double> naiveBayesTestInput = {2.5, 20.0};
    int naiveBayesPrediction = naiveBayes.predict(naiveBayesTestInput);
    std::cout << "Naive Bayes prediction for input {2.5, 20.0}: " << naiveBayesPrediction << std::endl;

    // Test for KNN
    std::cout << "\nK-Nearest Neighbors Model Example:" << std::endl;

    // Sample data for KNN
    // Format: {feature1, feature2, label}
    std::vector<std::vector<double>> knnTrainingData = {
        {1.0, 2.0, 0},
        {1.5, 1.5, 0},
        {5.0, 2.0, 1},
        {6.0, 2.0, 1},
        // ... add more data as needed
    };

    // Separate labels from features
    std::vector<std::vector<double>> knnData;
    std::vector<int> knnLabels;
    for (const auto &row : knnTrainingData)
    {
        knnData.push_back(std::vector<double>{row.begin(), row.end() - 1});
        knnLabels.push_back(static_cast<int>(row.back()));
    }

    KNN knn(3);                    // Initialize KNN with k=3
    knn.train(knnData, knnLabels); // Train the model

    // Test the KNN model with a sample input
    std::vector<double> knnTestInput = {2.0, 3.0};
    int knnPrediction = knn.predict(knnTestInput);
    std::cout << "KNN prediction for input {2.0, 3.0}: " << knnPrediction << std::endl;

    return 0;
}
