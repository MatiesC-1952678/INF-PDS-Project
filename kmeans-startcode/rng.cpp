#include "rng.h"
#include <cstdlib>
#include <iostream>

using namespace std;

Rng::Rng(unsigned long seed)
	: m_rng(seed), m_seed(seed)
{
}

Rng::~Rng()
{
}

// Algorithm from gsl_ran_choose (https://github.com/ampl/gsl/blob/master/randist/shuffle.c)
void Rng::pickRandomIndices(size_t n, std::vector<size_t> &indices)
{
	if (indices.size() > n)
	{
		cerr << "Requested more numbers than available" << endl;
		exit(-1);
	}

	for (size_t i = 0, j = 0 ; i < n && j < indices.size(); i++)
	{
		uniform_int_distribution<> dist(0, n-i-1);
		if ((size_t)dist(m_rng) < indices.size()-j)
		{
			indices[j] = i;
			j++;
		}
	}
}
