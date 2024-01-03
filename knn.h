#ifndef KNN_H
#define KNN_H

#include <vector>
#include <utility>

class KNN
{
private:
    int k;
    std::vector<std::vector<double>> trainData;
    std::vector<int> trainLabels;

    double euclideanDistance(const std::vector<double> &a, const std::vector<double> &b) const;

public:
    KNN(int k);
    void train(const std::vector<std::vector<double>> &data, const std::vector<int> &labels);
    int predict(const std::vector<double> &dataPoint) const;
};

#endif // KNN_H
