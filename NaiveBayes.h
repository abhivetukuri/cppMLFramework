#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <vector>
#include <map>

class NaiveBayes
{
private:
    struct Stats
    {
        double mean;
        double variance;
    };

    std::map<int, std::vector<Stats>> classStats;
    std::map<int, double> classPriors;
    int featureCount;

    Stats calculateStats(const std::vector<double> &data);
    double gaussianProbability(double x, double mean, double variance) const;

public:
    NaiveBayes();
    void train(const std::vector<std::vector<double>> &dataset);
    int predict(const std::vector<double> &data) const;
};

#endif // NAIVE_BAYES_H
