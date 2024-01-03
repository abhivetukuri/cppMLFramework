#include "NaiveBayes.h"
#include <cmath>
#include <numeric>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

NaiveBayes::NaiveBayes() : featureCount(0) {}

NaiveBayes::Stats NaiveBayes::calculateStats(const std::vector<double> &data)
{
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    for (double value : data)
    {
        variance += std::pow(value - mean, 2);
    }
    variance /= data.size();
    return {mean, variance};
}

double NaiveBayes::gaussianProbability(double x, double mean, double variance) const
{
    double exponent = std::exp(-(std::pow(x - mean, 2) / (2 * variance)));
    return (1 / std::sqrt(2 * M_PI * variance)) * exponent;
}

void NaiveBayes::train(const std::vector<std::vector<double>> &dataset)
{
    std::map<int, std::vector<std::vector<double>>> separated;
    featureCount = dataset[0].size() - 1;

    for (const auto &row : dataset)
    {
        separated[static_cast<int>(row.back())].push_back(row);
    }

    for (const auto &pair : separated)
    {
        int classValue = pair.first;
        const auto &rows = pair.second;
        std::vector<Stats> stats;
        for (int i = 0; i < featureCount; ++i)
        {
            std::vector<double> feature(rows.size());
            for (size_t j = 0; j < rows.size(); ++j)
            {
                feature[j] = rows[j][i];
            }
            stats.push_back(calculateStats(feature));
        }
        classStats[classValue] = stats;
        classPriors[classValue] = static_cast<double>(rows.size()) / dataset.size();
    }
}

int NaiveBayes::predict(const std::vector<double> &data) const
{
    std::map<int, double> probabilities;
    for (const auto &pair : classStats)
    {
        int classValue = pair.first;
        double classProbability = classPriors.at(classValue);
        const auto &stats = pair.second;

        probabilities[classValue] = classProbability;
        for (int i = 0; i < featureCount; ++i)
        {
            probabilities[classValue] *= gaussianProbability(data[i], stats[i].mean, stats[i].variance);
        }
    }

    int bestLabel = -1;
    double bestProb = -1;
    for (const auto &probability : probabilities)
    {
        if (bestLabel == -1 || probability.second > bestProb)
        {
            bestProb = probability.second;
            bestLabel = probability.first;
        }
    }

    return bestLabel;
}
