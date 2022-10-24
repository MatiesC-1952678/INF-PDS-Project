#pragma once
#include <vector>

class Point {
    public:
        Point(std::vector<int> &coo) : m_coordinates(coo) {}
        static void setDimensionForAll(int dim) {
            Point::dimension = dim;
        };
    private:
        std::vector<int> m_coordinates;
        static int dimension;
};

