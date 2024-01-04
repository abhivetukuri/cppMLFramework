#include "KMeans.h"
#include <limits>
#include <algorithm>
#include <map>
#include <random>

KMeans::KMeans(int k, int maxIterations) : k(k), maxIterations(maxIterations) {}

double KMeans::euclideanDistance(const std::vector<double> &a, const std::vector<double> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

std::vector<std::vector<double>> KMeans::initializeCentroids(const std::vector<std::vector<double>> &data)
{
    std::vector<std::vector<double>> centroids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    // Ensure k is not greater than the number of data points
    size_t actualK = std::min(static_cast<size_t>(k), data.size());

    std::map<int, bool> chosen;
    while (centroids.size() < actualK)
    {
        int idx = dis(gen);
        if (chosen.find(idx) == chosen.end())
        {
            centroids.push_back(data[idx]);
            chosen[idx] = true;
        }
    }
    return centroids;
}

int KMeans::closestCentroid(const std::vector<double> &point) const
{
    double minDistance = std::numeric_limits<double>::max();
    int centroidIndex = 0;

    for (int i = 0; i < k; ++i)
    {
        double distance = euclideanDistance(point, centroids[i]);
        if (distance < minDistance)
        {
            minDistance = distance;
            centroidIndex = i;
        }
    }

    return centroidIndex;
}

void KMeans::updateCentroids(const std::vector<std::vector<double>> &data, const std::vector<int> &assignments)
{
    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(data[0].size(), 0.0));
    std::vector<int> counts(k, 0);

    for (size_t i = 0; i < data.size(); ++i)
    {
        int cluster = assignments[i];
        for (size_t j = 0; j < data[i].size(); ++j)
        {
            newCentroids[cluster][j] += data[i][j];
        }
        counts[cluster]++;
    }

    for (int i = 0; i < k; ++i)
    {
        if (counts[i] > 0)
        {
            for (size_t j = 0; j < newCentroids[i].size(); ++j)
            {
                newCentroids[i][j] /= counts[i];
            }
        }
    }

    centroids = std::move(newCentroids);
}

void KMeans::train(const std::vector<std::vector<double>> &data)
{
    if (data.empty())
        return;

    centroids = initializeCentroids(data);

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        std::vector<int> assignments(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            assignments[i] = closestCentroid(data[i]);
        }

        updateCentroids(data, assignments);
    }
}

std::vector<int> KMeans::predict(const std::vector<std::vector<double>> &data) const
{
    std::vector<int> predictions;
    if (centroids.empty())
        return predictions;

    predictions.reserve(data.size());
    for (const auto &point : data)
    {
        predictions.push_back(closestCentroid(point));
    }
    return predictions;
}
