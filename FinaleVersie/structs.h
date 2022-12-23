#pragma once
#include <vector>
#include <stdio.h>
#include <memory>

struct point
{
	std::vector<double> datapoints;

	point() {
		datapoints = std::vector<double>();
	};

	~point(){};

	void addDataPoint(const double inputDataPoint)
	{
		datapoints.push_back(inputDataPoint);
	}

	double getDataPoint(const size_t index)
	{
		return datapoints[index];
	}

	size_t getSize()
	{
		return datapoints.size();
	}

	void add(point &p)
	{
		if (datapoints.size() <= 0)
			datapoints = std::vector<double>(p.getSize(), 0);

		for (size_t i = 0; i < p.getSize(); i++)
			datapoints[i] += p.getDataPoint(i);
	}

	void divide(const size_t divider)
	{
		for (size_t i = 0; i < datapoints.size(); i++)
			datapoints[i] /= divider;
	}

	double* toDoubleArray(){
		return datapoints.data();
	}
};
