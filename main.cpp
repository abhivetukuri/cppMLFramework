#include "Perceptron.h"
#include "LinearRegression.h"
#include "NaiveBayes.h"
#include "KNN.h"
#include "KMeans.h"
#include <iostream>
#include <vector>

void demoPerceptron();
void demoLinearRegression();
void demoNaiveBayes();
void demoKNN();
void demoKMeans();

int main()
{
    std::cout << "Welcome to the C++ ML Framework Demo\n";
    demoPerceptron();
    demoLinearRegression();
    demoNaiveBayes();
    demoKNN();
    demoKMeans();

    return 0;
}

void demoPerceptron()
{
    std::cout << "\n--- Perceptron Demo ---\n";
    std::vector<std::vector<double>> perceptronData = {
        {1.0, 2.0, 0.0},
        {1.5, 1.5, 0.0},
        {5.0, 2.0, 1.0}};
    Perceptron perceptron;
    perceptron.train(perceptronData, 0.1, 10);
    std::vector<double> perceptronTestInput = {2.0, 3.0};
    double perceptronPrediction = perceptron.predict(perceptronTestInput);
    std::cout << "Perceptron prediction for input {2.0, 3.0}: " << perceptronPrediction << std::endl;
}

void demoLinearRegression()
{
    std::cout << "\n--- Linear Regression Demo ---\n";
    std::vector<std::vector<double>> linearRegData = {
        {1.0, 2.0},
        {2.0, 3.0},
        {4.0, 5.0}};
    LinearRegression linearReg;
    linearReg.train(linearRegData, 0.01, 100);
    double linearRegTestInput = 3.0;
    double linearRegPrediction = linearReg.predict(linearRegTestInput);
    std::cout << "Linear Regression prediction for input 3.0: " << linearRegPrediction << std::endl;
}

void demoNaiveBayes()
{
    std::cout << "\n--- Naive Bayes Demo ---\n";
    std::vector<std::vector<double>> naiveBayesData = {
        {1.0, 20.0, 0},
        {2.0, 22.0, 0},
        {3.0, 18.0, 1},
        {2.0, 21.0, 1}};
    NaiveBayes naiveBayes;
    naiveBayes.train(naiveBayesData);
    std::vector<double> naiveBayesTestInput = {2.5, 20.0};
    int naiveBayesPrediction = naiveBayes.predict(naiveBayesTestInput);
    std::cout << "Naive Bayes prediction for input {2.5, 20.0}: " << naiveBayesPrediction << std::endl;
}

void demoKNN()
{
    std::cout << "\n--- K-Nearest Neighbors (KNN) Demo ---\n";
    std::vector<std::vector<double>> knnTrainingData = {
        {1.0, 2.0, 0},
        {1.5, 1.5, 0},
        {5.0, 2.0, 1},
        {6.0, 2.0, 1}};
    std::vector<std::vector<double>> knnData;
    std::vector<int> knnLabels;
    for (const auto &row : knnTrainingData)
    {
        knnData.push_back(std::vector<double>{row.begin(), row.end() - 1});
        knnLabels.push_back(static_cast<int>(row.back()));
    }
    KNN knn(3);
    knn.train(knnData, knnLabels);
    std::vector<double> knnTestInput = {2.0, 3.0};
    int knnPrediction = knn.predict(knnTestInput);
    std::cout << "KNN prediction for input {2.0, 3.0}: " << knnPrediction << std::endl;
}

void demoKMeans()
{
    std::cout << "\n--- K-Means Clustering Demo ---\n";
    std::vector<std::vector<double>> kmeansData = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}};
    int numClusters = 3;
    KMeans kmeans(numClusters, 100);
    kmeans.train(kmeansData);
    std::vector<int> clusters = kmeans.predict(kmeansData);
    for (size_t i = 0; i < clusters.size(); ++i)
    {
        std::cout << "Data point " << i << " is in cluster " << clusters[i] << std::endl;
    }
}
