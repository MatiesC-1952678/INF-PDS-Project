#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <math.h> /* floor */
#include <cuda.h>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"
#include "structs.h"
#include <algorithm>

/* CUDA KERNELS */
__global__ void updateCentroids(){

}

/*
	Find the closest centroids index and distance for a given point
	@param numClusters: number of centroids needed
	@param dist: the distance to the closest centroid
	@param centroids: a vector of random points from the input file
	@param offset: the offset in the vectors to seperate repetitions
	@pre dist is infinity
	@post dist is the distance to the closest centroid
	@return the index of the closest centroid
*/
__device__ int find_closest_centroid_index_and_distance(double &dist, double *p, double *centroids, const int numClusters, const size_t offset, const int dimension)
{
	// point closestCentroid;
	int indexCentroid;
	for (size_t c = 0; c < numClusters; ++c)
	{
		double currentdist = 0;
		int whichCentroid = c * dimension;

		for (size_t i = 0; i < dimension; ++i)
		{ // p.getSize() or dimension = N
			currentdist += pow((p[i] - centroids[offset + whichCentroid + i]), 2);
			//printf("%d: %f - whichCentroid: %d \n",c, currentdist, whichCentroid);
			// printf("p: %f\n", p[i]);
			// printf("c: %f - offset:%d - i: %d\n", centroids[offset + whichCentroid + i], offset, i);
		}

		if (dist == -1)
		{
			// closestCentroid = centroids[offset + c];
			dist = currentdist;
			indexCentroid = c;
		}
		else if (currentdist < dist)
		{
			// closestCentroid = centroids[offset + c];
			dist = currentdist;
			indexCentroid = c;
		}
	}
	return indexCentroid;
}

/*
TODO:
	- double* distanceSquaredSum
	- threadRange and threadSurplus implementation (now assuming threadRange is constant)
	- assume repetitions is part of a 2D matrix (so you can also do repetitions in parallel)
*/
__global__ void assignNewClusters(
	int *cuClusters,
	const size_t clusterOffset,
	double *cuCentroids,
	const size_t centroidOffset,
	double *cuPoints,
	const int threadRange,
	const int numClusters,
	double *distanceSquaredSum,
	bool *cuChanged,
	const int dimension,
	const int numPoints)
{
	/*
		1 rep =
		blocks -> 	|	|	|	|	| * |	|	|	|	|	|
						threads ->	//|\\
									12345
									  |
									  v
									{...}
		blockIdx = 		which block am I?
		blockDim = 		how big is this certain block?
		threadIdx = 	which thread am I in block *?
		threadRange = 	how many datapoints does this thread get?

		ex: blockIdx = *, blockDim = len(*), threadIdx = 3, threadRange = len({...})
	*/

	int start = 0; // blockIdx.x * blockDim.x  + (threadIdx.x * threadRange)
  printf("%d", threadIdx.x);
	int stop = start + threadRange;
	for (int p = start; p < numPoints; ++p)
	{
		double dist = -1;
		const int newCluster = find_closest_centroid_index_and_distance(dist, &cuPoints[p * dimension], cuCentroids, numClusters, centroidOffset, dimension);
		*distanceSquaredSum += dist; // REDUCTION

		if (newCluster != cuClusters[clusterOffset + p * dimension])
		{
			cuClusters[clusterOffset + p * dimension] = newCluster;
			*cuChanged = true;
		}
	}
}

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
void readData(std::ifstream &input, std::vector<double> &allData, size_t &numRows, size_t &numCols)
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

		for (auto x : row)
		{
			allData.push_back(x);
		}

		line++;
	}

	numRows = (size_t)allData.size() / numColsExpected;
	numCols = (size_t)numColsExpected;
}

// Helper function to open debug file
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

/*
	Chooses random points from the input file to use as centroids
	@param numClusters: number of centroids needed
	@param rng: class to retrieve random indices
	@param centroids: a vector of random points from the input file
	@param repetitions: the number of repetitions
	@param allPoints: all the points from the input file
	@pre centroids vector is empty
	@post centroids is filled with random points
*/
void choose_centroids_at_random(const int numClusters, const int dimension, Rng &rng, std::vector<double> &centroids, const int repetitions, std::vector<double> &allPoints)
{
	for (int rep = 0; rep < repetitions; ++rep)
	{
		std::vector<size_t> indices(numClusters);
		rng.pickRandomIndices(allPoints.size() / dimension, indices);

		const int whichRep = numClusters * rep * dimension;

		for (size_t cluster = 0; cluster < numClusters; cluster++)
		{
			const int whichCluster = cluster * dimension;

			// printf("%d\n", indices[cluster]);
			for (size_t whichCoordinate = 0; whichCoordinate < dimension; whichCoordinate++)
			{
				centroids[whichRep + whichCluster + whichCoordinate] = allPoints[indices[cluster] * dimension + whichCoordinate];
				// printf("centroids%d %f\n", whichCoordinate, allPoints[indices[cluster]*dimension + whichCoordinate]);
			}
		}
	}

	// for (size_t i = 0; i < allPoints.size() ; i++)
	// {
	// printf("%f\n", allPoints[i]);
	// printf("%f\n", centroids[i]);
	// }
}

/*
	Average of all points from a given cluster
	@param centroidIndex: the index of the centroid
	@param clusters: vector with the closest centroid per point
	@param clusterOffset: the offset in the cluster vector to seperate repetitions
	@param allPoints: all the points from the input file
	@return the average point from the cluster
*/
void average_of_points_with_cluster(
	double *avgPoint,
	const size_t centroidIndex,
	int *cuClusters,
	const size_t clusterOffset,
	double *allPoints,
	const int numPoints,
	const int dimension)
{
	size_t numberOfPoints = 0;
	for (size_t p = 0; p < numPoints; p++)
	{
		if (cuClusters[clusterOffset + p * dimension] == centroidIndex)
		{
			for (size_t i = 0; i < dimension; i++)
			{
				avgPoint[i] += allPoints[p * dimension + i];
				//printf("%f\n", avgPoint[i]);
			}
			numberOfPoints++;
			//printf("--- number of points  %d\n", numberOfPoints);
		}
	}
	// std::cout << avgPoint.getDataPoint(0);
	for (size_t i = 0; i < dimension; i++)
		avgPoint[i] /= numberOfPoints;
}

/*
	Writes the clusters to the debug file
	@param numpoints: total number of points
	@param cluster: vector with the closest centroid per point
	@param clusterDebugFileName: the file name of the debug file
*/
void writeClusterToDebugFile(std::vector<double> &cluster, std::string &clusterDebugFileName, const int numpoints)
{
	FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName);
	clustersDebugFile.write(cluster, numpoints);
	clustersDebugFile.close();
}

/*
	Writes the centroids to the debug file
	@param dimension: dimenion of the points
	@param centroid: vector with the closest centroid per point
	@param centroidDebugFileName: the file name of the debug file
*/
void writeCentroidToDebugFile(std::vector<double> &centroid, std::string &centroidDebugFileName, const int dimension)
{
	FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName);
	centroidDebugFile.write(centroid, dimension);
	centroidDebugFile.close();
}

/*
	Does a kmeans run
	@param bestDistSquaredSum: the best distance
	@param bestClusterOffset: the offset of the best cluster
	@param centroids: a vector of random points from the input file
	@param centroidOffset: the offset of the current centroids in this rep
	@param clusters: vector with the closest centroid per point
	@param clusterOffset: the offset of the current cluster in this rep
	@param allPoints: all the points from the input file
	@param numPoints: the total number of points
	@param numClusters: number of centroids needed
	@param debugCentroids: centroidtrace flag is set
	@param debugClusters: trace flag is set
	@param centroidDebugFile: centroids debug file
	@param clustersDebugFile: clusters debug file
	@return the amount of steps this run took to complete
 */
int kmeansReps(double &bestDistSquaredSum,
			   size_t &bestClusterOffset,
			   double *cuCentroids,
			   size_t centroidOffset,
			   int *cuClusters,
			   size_t clusterOffset,
			   double *cuPoints,
			   bool *cuChanged,
			   double *cuDistanceSquaredSum,
			   const size_t numPoints,
			   const int numClusters,
			   const int numBlocks,
			   const int numThreads,
			   bool debugCentroids,
			   bool debugClusters,
			   std::string &centroidDebugFile,
			   std::string &clustersDebugFile,
			   const int dimension)
{
	int steps = 0;
	// std::vector<double> debugCluster{};
	// std::vector<double> debugCentroid{};
  std::cout << (*cuChanged);

	// while (*cuChanged)
	// {
    

		// steps++;
		// *cuChanged = false;
		// *cuDistanceSquaredSum = 0.0;
    printf("test");

		// CUDA: Wordt Cuda kernel
		// TODO: overschot verdelen over alle blocks (niet enkel de laatste)
		// int blockRange = floor(numPoints / numBlocks);
		// int surplusBlocks = numPoints % numBlocks;
		// int threadRange = floor(blockRange / numThreads);
		// int surplusThreads = blockRange % numThreads;
		// assignNewClusters<<<32,32>>>(
		// 	cuClusters,
		// 	clusterOffset,
		// 	cuCentroids,
		// 	centroidOffset,
		// 	cuPoints,
		// 	threadRange,
		// 	numClusters,
		// 	cuDistanceSquaredSum,
		// 	cuChanged,
		// 	dimension,
		// 	(int)numPoints);

		// cudaMemcpy(&changed, cuChanged, sizeof(bool), cudaMemcpyDeviceToHost);

		// if (debugClusters)
		// 	debugCluster.insert(debugCluster.end(), &clusters[0], &clusters[numPoints]);
		// if (debugCentroids)
		// {
		// 	for (size_t whichCoordinate = 0; whichCoordinate < numClusters; whichCoordinate++)
		// 	{
		// 		if (debugCentroids)
		// 		{
		// 			for (size_t whichCluster = 0; whichCluster < allPoints[0].getSize(); whichCluster++)
		// 				debugCentroid.push_back(centroids[whichCoordinate].getDataPoint(whichCluster));
		// 		}
		// 	}
		// }

		// 2. averages
		// if (*cuChanged)
		// {
		// 	// memCopy 1 average point
		// 	//  CUDA: Wordt Cuda kernel
		// 	for (size_t cluster = 0; cluster < numClusters; ++cluster)
		// 	{
		// 		std::vector<double> averagePoint(dimension, 0);
		// 		average_of_points_with_cluster(&averagePoint[0], cluster, cuClusters, clusterOffset, cuPoints, numPoints, dimension);

		// 		//printf("--- centroids \n");
		// 		for (int coor = 0; coor < dimension; coor++)
		// 		{
		// 			cuCentroids[centroidOffset + cluster * dimension + coor] = averagePoint[coor];
		// 			//printf("%f\n", cuCentroids[centroidOffset + cluster * dimension + coor]);
		// 		}
		// 	}
		// }

		// if (*cuDistanceSquaredSum < bestDistSquaredSum)
		// {
		// 	bestClusterOffset = clusterOffset;
		// 	bestDistSquaredSum = *cuDistanceSquaredSum;
		// }

		//exit(0);
	// }

	// if (debugClusters)
	// 	writeClusterToDebugFile(debugCluster, clustersDebugFile, numPoints);
	// if (debugCentroids)
	// 	writeCentroidToDebugFile(debugCentroid, centroidDebugFile, allPoints[0].getSize());

	return steps;
}

int kmeans(Rng &rng,
		   const std::string inputFile,
		   const std::string &outputFileName,
		   int numClusters,
		   int repetitions,
		   int numBlocks,
		   int numThreads,
		   std::string &centroidDebugFileName,
		   std::string &clusterDebugFileName)
{

	FileCSVWriter csvOutputFile(outputFileName);
	if (!csvOutputFile.is_open())
	{
		std::cerr << "Unable to open output file " << outputFileName << std::endl;
		return -1;
	}

	static size_t numPoints = 0; // Will reset in the readData function anyways = rows
	static size_t dimension = 0; // Will reset in the readData function anyways = cols
	std::vector<double> allPoints{};
	std::ifstream infile(inputFile);
	readData(infile, allPoints, numPoints, dimension);

	// THEN start timing! (don't want to also time the creation of our big variables, make it upfront and CONSTANT overhead)
	// This is a basic timer from std::chrono ; feel free to use the appropriate timer for
	// each of the technologies, e.g. OpenMP has omp_get_wtime()
	Timer timer;

	// initialize BIG variabels
	size_t bestClusterOffset{0};
	/**
	 * with amount of -1 in the matrix = number of points * repetitions
	 * abstract example: [-1, -1, | -1, -1, | -1, -1]
	 * the | (abstract, not actually in the vector) defines a new repetition,
	 * in this case there are only 2 points in the entire csv file (2 rows),
	 * and 3 repetitions in total
	 */
	std::vector<int> clusters((int)numPoints * repetitions, -1);

	/**
	 * centroids * repetitions
	 * abstract example: [p1, p2, p3 | p1, p2, p3]
	 * here are 3 centroids for 3 repetitions made
	 */
	std::vector<double> centroids(numClusters * repetitions * dimension);

	choose_centroids_at_random(numClusters, dimension, rng, centroids, repetitions, allPoints);

	double bestDistSquaredSum = std::numeric_limits<double>::max(); // can only get better
	std::vector<size_t> stepsPerRepetition(repetitions);			// to save the number of steps each rep needed

	bool changed = true;
	// std::vector<double> distanceSquaredSum(repetitions, 0);
	double distanceSquaredSum = 0.0;

	// CUDA: CPU -> GPU allocation
	int *cuClustersPointer = &clusters[0];
	double *cuCentroidsPointer = &centroids[0];
	double *cuPointsPointer = &allPoints[0];
	bool* cuChangedPointer;
	double *cuDistanceSquaredSumPointer = &distanceSquaredSum;

	size_t sizeOfClusters = numPoints * repetitions * sizeof(int);
	size_t sizeOfCentroids = numClusters * repetitions * dimension * sizeof(double);
	size_t sizeOfPoints = numPoints * dimension * sizeof(double);
	size_t sizeOfChanged = sizeof(bool);
	size_t sizeOfDistanceSquaredSum = sizeof(double);

	cudaMalloc(&cuClustersPointer, sizeOfClusters);
	cudaMalloc(&cuCentroidsPointer, sizeOfCentroids);
	cudaMalloc(&cuPointsPointer, sizeOfPoints);
	cudaMalloc(&cuChangedPointer, sizeOfChanged);
	cudaMalloc(&cuDistanceSquaredSumPointer, sizeOfDistanceSquaredSum);

	cudaMemcpy(cuClustersPointer, clusters.data(), sizeOfClusters, cudaMemcpyHostToDevice);
	cudaMemcpy(cuCentroidsPointer, centroids.data(), sizeOfCentroids, cudaMemcpyHostToDevice);
	cudaMemcpy(cuPointsPointer, allPoints.data(), sizeOfPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(cuChangedPointer, &changed, sizeOfChanged, cudaMemcpyHostToDevice);
	cudaMemcpy(cuDistanceSquaredSumPointer, &distanceSquaredSum, sizeOfDistanceSquaredSum, cudaMemcpyHostToDevice);

	// // Do the k-means routine a number of times, each time starting from
	// // different random centroids (use Rng::pickRandomIndices), and keep
	// // the best result of these repetitions.

	for (int r = 0; r < repetitions; r++)
	{
		size_t numSteps = 0;

		// printf("Rep - Thread %d\n", 5);

		// if (centroidDebugFileName.length() > 0 && clusterDebugFileName.length() > 0 && r == 0) {
		stepsPerRepetition[r] = kmeansReps(
			bestDistSquaredSum,
			bestClusterOffset,
			cuCentroidsPointer,			 // CUDA centroids pointer
			numClusters * r * dimension, // centroids internal offset for this rep
			cuClustersPointer,			 // CUDA clusters pointer
			numPoints * r * dimension,	 //
			cuPointsPointer,
			cuChangedPointer,
			cuDistanceSquaredSumPointer,
			numPoints,
			numClusters,
			numBlocks,
			numThreads,
			false,
			false,
			centroidDebugFileName,
			clusterDebugFileName,
			dimension);
		//}
		// else if (centroidDebugFileName.length() > 0 && r == 0)
		// 	stepsPerRepetition[r] = kmeansReps(bestDistSquaredSum, bestClusterOffset, centroids, numClusters * r, clusters, numPoints * r, allPoints, numPoints, numClusters, true, false, centroidDebugFileName, clusterDebugFileName);
		// else if (clusterDebugFileName.length() > 0 && r == 0)
		// 	stepsPerRepetition[r] = kmeansReps(bestDistSquaredSum, bestClusterOffset, centroids, numClusters * r, clusters, numPoints * r, allPoints, numPoints, numClusters, false, true, centroidDebugFileName, clusterDebugFileName);
		// else
		// 	stepsPerRepetition[r] = kmeansReps(bestDistSquaredSum, bestClusterOffset, centroids, numClusters * r, clusters, numPoints * r, allPoints, numPoints, numClusters, false, false, centroidDebugFileName, clusterDebugFileName);
	}
	// TODO: CUDA: GPU -> CPU allocation + cudaFree

	timer.stop();

  // cudaFree(cuClustersPointer);
	// cudaFree(cuCentroidsPointer);
	// cudaFree(cuPointsPointer);
	// cudaFree(cuChangedPointer);
	// cudaFree(cuDistanceSquaredSumPointer);

	// Some example output, of course you can log your timing data anyway you like.
	std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
	std::cout << "sequential," << numBlocks << "," << numThreads << "," << inputFile << ","
			  << rng.getUsedSeed() << "," << numClusters << ","
			  << repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds() / 1e9
			  << std::endl;

	// Write the number of steps per repetition, kind of a signature of the work involved
	csvOutputFile.write(stepsPerRepetition, "# Steps: ");
	// Write best clusters to csvOutputFile, something like

	// Get best cluster from repetitions -> example output shows the first cluster, we use the best cluster for output
	std::vector<int> bestCluster(&clusters[bestClusterOffset], &clusters[bestClusterOffset + numPoints]);
	csvOutputFile.write(bestCluster);
	return 0;
}

int mainCxx(const std::vector<std::string> &args)
{
	if (args.size() % 2 != 0)
		usage();

	std::string inputFileName, outputFileName, centroidTraceFileName, clusterTraceFileName;
	unsigned long seed = 0;

	int numClusters = -1;
	int repetitions = -1;
	int numBlocks = 1, numThreads = 1;
	for (int whichCluster = 0; whichCluster < args.size(); whichCluster += 2)
	{
		if (args[whichCluster] == "--input")
			inputFileName = args[whichCluster + 1];
		else if (args[whichCluster] == "--output")
			outputFileName = args[whichCluster + 1];
		else if (args[whichCluster] == "--centroidtrace")
			centroidTraceFileName = args[whichCluster + 1];
		else if (args[whichCluster] == "--trace")
			clusterTraceFileName = args[whichCluster + 1];
		else if (args[whichCluster] == "--k")
			numClusters = stoi(args[whichCluster + 1]);
		else if (args[whichCluster] == "--repetitions")
			repetitions = stoi(args[whichCluster + 1]);
		else if (args[whichCluster] == "--seed")
			seed = stoul(args[whichCluster + 1]);
		else if (args[whichCluster] == "--blocks")
			numBlocks = stoi(args[whichCluster + 1]);
		else if (args[whichCluster] == "--threads")
			numThreads = stoi(args[whichCluster + 1]);
		else
		{
			std::cerr << "Unknown argument '" << args[whichCluster] << "'" << std::endl;
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
	for (int whichCluster = 1; whichCluster < argc; whichCluster++)
		args.push_back(argv[whichCluster]);

	return mainCxx(args);
}
