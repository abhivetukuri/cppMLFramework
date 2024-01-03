#include "KNN.h"
#include <cmath>
#include <algorithm>
#include <map>

KNN::KNN(int k) : k(k) {}

void KNN::train(const std::vector<std::vector<double>> &data, const std::vector<int> &labels)
{
    trainData = data;
    trainLabels = labels;
}

double KNN::euclideanDistance(const std::vector<double> &a, const std::vector<double> &b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}

int KNN::predict(const std::vector<double> &dataPoint) const
{
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < trainData.size(); ++i)
    {
        double dist = euclideanDistance(dataPoint, trainData[i]);
        distances.push_back(std::make_pair(dist, trainLabels[i]));
    }

    std::sort(distances.begin(), distances.end());

    std::map<int, int> classVotes;
    for (int i = 0; i < k; ++i)
    {
        classVotes[distances[i].second]++;
    }

    int maxVote = 0, predictedClass = -1;
    for (const auto &vote : classVotes)
    {
        if (vote.second > maxVote)
        {
            maxVote = vote.second;
            predictedClass = vote.first;
        }
    }

    return predictedClass;
}
