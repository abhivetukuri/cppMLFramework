#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>

class KMeans
{
private:
    int k;
    int maxIterations;
    std::vector<std::vector<double>> centroids;

    static double euclideanDistance(const std::vector<double> &a, const std::vector<double> &b);
    int closestCentroid(const std::vector<double> &point) const;
    std::vector<std::vector<double>> initializeCentroids(const std::vector<std::vector<double>> &data);
    void updateCentroids(const std::vector<std::vector<double>> &data, const std::vector<int> &assignments);

public:
    KMeans(int k, int maxIterations);
    void train(const std::vector<std::vector<double>> &data);
    std::vector<int> predict(const std::vector<std::vector<double>> &data) const;
};

#endif // KMEANS_H
