#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"

// TODO@yarne: maybe put this in a seperate file (like my structs.h file)
struct point
{
	std::vector<double> datapoints;

	point(){};

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
		for (size_t i = 0; i < p.getSize(); i++)
			datapoints[i] += p.getDataPoint(i);
	}

	void divide(const size_t divider)
	{
		for (size_t i = 0; i < datapoints.size(); i++)
			datapoints[i] /= divider;
	}
};

typedef std::vector<std::vector<point>> PointMatrix;
// TODO@ties: change vector to array for number of centroids, NOT for amount of steps -> cuz these are variable, repetitions are known
// typedef std::vector<std::vector<int>> IntMatrix;

void usage()
{
	std::cerr << R"XYZ(
Usage:

  kmeans --input inputfile.csv --output outputfile.csv --k numclusters --repetitions numrepetitions --seed seed [--blocks numblocks] [--threads numthreads] [--trace clusteridxdebug.csv] [--centroidtrace centroiddebug.csv]

Arguments:

 --input:
 
   Specifies input CSV file, number of rows represents number of points, the
   number of columns is the dimension of each point.

 --output:

   Output CSV file, just a single row, with as many entries as the number of
   points in the input file. Each entry is the index of the cluster to which
   the point belongs. The script 'visualize_clusters.py' can show this final
   clustering.

 --k:

   The number of clusters that should be identified.

 --repetitions:

   The number of times the k-means algorithm is repeated; the best clustering
   is kept.

 --blocks:

   Only relevant in CUDA version, specifies the number of blocks that can be
   used.

 --threads:

   Not relevant for the serial version. For the OpenMP version, this number 
   of threads should be used. For the CUDA version, this is the number of 
   threads per block. For the MPI executable, this should be ignored, but
   the wrapper script 'mpiwrapper.sh' can inspect this to run 'mpirun' with
   the correct number of processes.

 --seed:

   Specifies a seed for the random number generator, to be able to get 
   reproducible results.

 --trace:

   Debug option - do NOT use this when timing your program!

   For each repetition, the k-means algorithm goes through a sequence of 
   increasingly better cluster assignments. If this option is specified, this
   sequence of cluster assignments should be written to a CSV file, similar
   to the '--output' option. Instead of only having one line, there will be
   as many lines as steps in this sequence. If multiple repetitions are
   specified, only the results of the first repetition should be logged
   for clarity. The 'visualize_clusters.py' program can help to visualize
   the data logged in this file.

 --centroidtrace:

   Debug option - do NOT use this when timing your program!

   Should also only log data during the first repetition. The resulting CSV 
   file first logs the randomly chosen centroids from the input data, and for
   each step in the sequence, the updated centroids are logged. The program 
   'visualize_centroids.py' can be used to visualize how the centroids change.
   
)XYZ";
	exit(-1);
}

// Helper function to read input file into allData, setting number of detected
// rows and columns. Feel free to use, adapt or ignore
void readData(std::ifstream &input, std::vector<point> &allData, size_t &numRows, size_t &numCols)
{
	if (!input.is_open())
		throw std::runtime_error("Input file is not open");

	allData.resize(0);
	numRows = 0;
	numCols = -1;

	CSVReader inReader(input);
	int numColsExpected = -1;
	int line = 1;
	std::vector<double> row;

	while (inReader.read(row))
	{
		if (numColsExpected == -1)
		{
			numColsExpected = row.size();
			if (numColsExpected <= 0)
				throw std::runtime_error("Unexpected error: 0 columns");
		}
		else if (numColsExpected != (int)row.size())
			throw std::runtime_error("Incompatible number of colums read in line " + std::to_string(line) + ": expecting " + std::to_string(numColsExpected) + " but got " + std::to_string(row.size()));

		point p{};
		for (auto x : row)
		{
			p.addDataPoint(x);
		}
		allData.push_back(p);

		line++;
	}

	numRows = (size_t)allData.size() / numColsExpected;
	numCols = (size_t)numColsExpected;
}

FileCSVWriter openDebugFile(const std::string &n)
{
	FileCSVWriter f;

	if (n.length() != 0)
	{
		f.open(n);
		if (!f.is_open())
			std::cerr << "WARNING: Unable to open debug file " << n << std::endl;
	}
	return f;
}

// PSEUDO CODE
// bestClusters = None
// bestDistanceSquaredSum = Infinity
//
// for r in range(repetitions):
// 		centroids = choose_centroids_at_random(k) # (use Rng to pick k random points)
// 		clusters = [ -1, ..., -1 ] 	# initially we don't know the closest centroid index
// 									# for each point
// 		changed = True
// 		while changed:
// 			changed = False
// 			distanceSquaredSum = 0
//
// 			for p in range(numberOfPoints)
// 				newCluster, dist = find_closest_centroid_index_and_distance(p, centroids)
// 				distanceSquaredSum += dist
//
//				if newCluster != clusters[p]:
//					clusters[p] = newCluster
//					changed = True
//
//			if changed: # re-calculate the centroids based on current clustering
//				for j in range(k):
//					centroids[j] = average_of_points_with_cluster(j)
//
//			# Keep track of best clustering
//			if distanceSquaredSum < bestDistanceSquaredSum:
//				bestClusters = clusters
//				bestDistanceSquaredSum = distanceSquaredSum
//

// Point choose_centroids_at_random(const pointMatrix& centroids, const int k);

std::vector<size_t> choose_centroids_at_random(const int k, Rng &rng)
{
	std::vector<size_t> centroids(k);
	rng.pickRandomIndices(k, centroids);
	std::cout << centroids[2];
	return centroids;
}

// TODO@yarne: Pretty sure dat pick random indices niet indices neemt van de totale punt vector ma gewoon random coordinaten maakt voor een nieuw centroid punt
// DIT MOET GE DUS WSS AANPASSEN
void choose_centroids_at_random(const int numClusters, const int numPoints, Rng &rng, std::vector<point> &centroids, std::vector<point> &allPoints)
{
	std::vector<size_t> indices(numClusters);
	rng.pickRandomIndices(numPoints, indices);
	for (size_t i = 0; i < numClusters; i++)
		centroids[i] = allPoints[indices[i]];
}

// int find_closest_centroid_index_and_distance(float &dist, const int p, std::vector<Point>& centroids)

// TODO@yarne: I would remove the -1 special case and just make the distance MAXVALUE
int find_closest_centroid_index_and_distance(float &dist, point &p, std::vector<point> &centroids)
{
	point closestCentroid;
	int indexCentroid;
	for (size_t c = 0; c < centroids.size(); c++)
	{
		double currentdist = 0;
		for (size_t i = 0; i < p.getSize() - 1; i++) // p.getSize() or dimension
		{
			currentdist += pow((p.getDataPoint(i) - centroids[c].getDataPoint(i)), 2);
		}
		if (dist == -1)
		{
			closestCentroid = centroids[c];
			dist = currentdist;
			indexCentroid = c;
		}
		else if (dist > currentdist)
		{
			closestCentroid = centroids[c];
			dist = currentdist;
			indexCentroid = c;
		}
	}
	return indexCentroid;
}

// //float or int???
// Point average_of_points_with_cluster(const int j)

point average_of_points_with_cluster(const size_t centroidID, const std::vector<int> &clusters, std::vector<point> &allPoints)
{
	// cluster -> punt -> welke centroid
	point avgPoint;
	size_t numberOfPoints = 0;
	for (size_t i = 0; i < clusters.size(); i++)
	{
		if (clusters[i] == centroidID)
		{
			if (i == 1)
				avgPoint = allPoints[i];
			else
				avgPoint.add(allPoints[i]);
			numberOfPoints++;
		}
	}
	avgPoint.divide(numberOfPoints);
	return avgPoint;
}

/**
 * possible optimizations:
 * 	- use best cluster, smallest integer representative (uchar f.e.) which means there can be max 255 clusters
 */
int kmeansReps(double &bestDistSquaredSum,
			   std::vector<int> *bestClusters,
			   std::vector<point> &centroids,
			   std::vector<int> &clusters,
			   std::vector<point> &allPoints,
			   const size_t numPoints,
			   const int numClusters			)
{

	bool changed = true;
	while (changed)
	{
		changed = false;
		float distanceSquaredSum = 0;

		for (int p = 0; p < numPoints; ++p)
		{
			float dist{-1};
			const int newCluster = find_closest_centroid_index_and_distance(dist, allPoints[p], centroids);
			distanceSquaredSum += dist;

			if (newCluster != clusters[p])
			{							  // optim: comparer that's quick?
				clusters[p] = newCluster; // optim: copy constructor? -> make a move constructor? -> which is faster?
				changed = true;
			}
		}

		if (changed)
		{
			for (int j = 0; j < numClusters; ++j)
			{
				centroids[j] = average_of_points_with_cluster(j, clusters, allPoints);
			}
		}

		if (distanceSquaredSum < bestDistSquaredSum)
		{
			bestClusters = &clusters; // Does this work with pointers?
			bestDistSquaredSum = distanceSquaredSum;
		}
	}
}

int kmeans(Rng &rng,
		   const std::string &inputFile,
		   const std::string &outputFileName,
		   int numClusters,
		   int repetitions,
		   int numBlocks,
		   int numThreads,
		   const std::string &centroidDebugFileName,
		   const std::string &clusterDebugFileName)
{
	// If debug filenames are specified, this opens them. The is_open method
	// can be used to check if they are actually open and should be written to.
	FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName);
	FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName);

	FileCSVWriter csvOutputFile(outputFileName);
	if (!csvOutputFile.is_open())
	{
		std::cerr << "Unable to open output file " << outputFileName << std::endl;
		return -1;
	}

	static size_t numPoints = 0; // Will reset in the readData function anyways = rows
	static size_t dimension = 0; // Will reset in the readData function anyways = cols
	std::vector<point> allPoints{};
	std::ifstream infile(inputFile);
	readData(infile, allPoints, numPoints, dimension);

	// initialize BIG variabels
	std::vector<int> *bestClusters{}; // TODO@ties: shouldn't be a pointer, should be an offset in vector
	std::vector<int> clusters ((int)numPoints * repetitions, -1);
	/**
	 * with amount of -1 in the matrix = number of points * repetitions
	 * abstract example: [-1, -1, | -1, -1, | -1, -1]
	 * the | (abstract, not actually in the vector) defines a new repetition,
	 * in this case there are only 2 points in the entire csv file (2 rows),
	 * and 3 repetitions in total
	 */

	PointMatrix centroids{repetitions};
	for (int i = 0; i < repetitions; ++i)
	{
		choose_centroids_at_random(numClusters, numPoints, rng, centroids[repetitions], allPoints);
	}
	double bestDistSquaredSum = std::numeric_limits<double>::max(); // can only get better
	std::vector<size_t> stepsPerRepetition(repetitions);			// to save the number of steps each rep needed

	// THEN start timing! (don't want to also time the creation of our big variables, make it upfront and CONSTANT overhead)
	// This is a basic timer from std::chrono ; feel free to use the appropriate timer for
	// each of the technologies, e.g. OpenMP has omp_get_wtime()
	Timer timer;

	// Do the k-means routine a number of times, each time starting from
	// different random centroids (use Rng::pickRandomIndices), and keep
	// the best result of these repetitions.
	for (int r = 0; r < repetitions; r++)
	{
		size_t numSteps = 0;

		// TODO: perform an actual k-means run, starting from random centroids
		//       (see rng.h)
		// std::cerr << "TODO: implement this" << std::endl;

		kmeansReps(bestDistSquaredSum, bestClusters, centroids[r], clusters, allPoints, numPoints, numClusters);
		// TODO@ties: change numPoints & numClusters & num reps from a non hardcoded var!

		stepsPerRepetition[r] = numSteps;

		// Make sure debug logging is only done on first iteration ; subsequent checks
		// with is_open will indicate that no logging needs to be done anymore.
		centroidDebugFile.close();
		clustersDebugFile.close();
	}

	timer.stop();

	// Some example output, of course you can log your timing data anyway you like.
	std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
	std::cout << "sequential," << numBlocks << "," << numThreads << "," << inputFile << ","
			  << rng.getUsedSeed() << "," << numClusters << ","
			  << repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds() / 1e9
			  << std::endl;

	// Write the number of steps per repetition, kind of a signature of the work involved
	csvOutputFile.write(stepsPerRepetition, "# Steps: ");
	// Write best clusters to csvOutputFile, something like
	// csvOutputFile.write( best cluster indices )
	return 0;
}

int mainCxx(const std::vector<std::string> &args)
{
	if (args.size() % 2 != 0)
		usage();

	std::string inputFileName, outputFileName, centroidTraceFileName, clusterTraceFileName;
	unsigned long seed = 0;

	int numClusters = -1, repetitions = -1;
	int numBlocks = 1, numThreads = 1;
	for (int i = 0; i < args.size(); i += 2)
	{
		if (args[i] == "--input")
			inputFileName = args[i + 1];
		else if (args[i] == "--output")
			outputFileName = args[i + 1];
		else if (args[i] == "--centroidtrace")
			centroidTraceFileName = args[i + 1];
		else if (args[i] == "--trace")
			clusterTraceFileName = args[i + 1];
		else if (args[i] == "--k")
			numClusters = stoi(args[i + 1]);
		else if (args[i] == "--repetitions")
			repetitions = stoi(args[i + 1]);
		else if (args[i] == "--seed")
			seed = stoul(args[i + 1]);
		else if (args[i] == "--blocks")
			numBlocks = stoi(args[i + 1]);
		else if (args[i] == "--threads")
			numThreads = stoi(args[i + 1]);
		else
		{
			std::cerr << "Unknown argument '" << args[i] << "'" << std::endl;
			return -1;
		}
	}

	if (inputFileName.length() == 0 || outputFileName.length() == 0 || numClusters < 1 || repetitions < 1 || seed == 0)
		usage();

	Rng rng(seed);

	return kmeans(rng, inputFileName, outputFileName, numClusters, repetitions,
				  numBlocks, numThreads, centroidTraceFileName, clusterTraceFileName);
}

int main(int argc, char *argv[])
{
	std::vector<std::string> args;
	for (int i = 1; i < argc; i++)
		args.push_back(argv[i]);

	return mainCxx(args);
}
