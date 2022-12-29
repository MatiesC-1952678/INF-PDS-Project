#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"
#include "structs.h"
#include "mpi.h"

int mpi_size;
int mpi_rank;
const static int mpi_rootRank = 0;

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
		rng.pickRandomIndices(allPoints.size()/dimension, indices);
		const int whichRep = numClusters * rep * dimension;
		for (size_t cluster = 0; cluster < numClusters; cluster++){
			const int whichCluster = cluster * dimension;
			// printf("Rep: %d, Cluster: %d, Centroid: %f,%f\n",rep,cluster,allPoints[indices[cluster] * dimension], allPoints[indices[cluster] * dimension + 1]);
			for (size_t coordinate = 0; coordinate < dimension; coordinate++)
				centroids[whichRep + whichCluster + coordinate] = allPoints[indices[cluster] * dimension + coordinate];
		}
	}
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
int find_closest_centroid_index_and_distance(double &dist, double *p, std::vector<double> &centroids, const int numClusters, const size_t offset, const int dimension)
{
 	// printf("Proces %d with offset %d\n",mpi_rank, offset);
	int indexCentroid;
	for (size_t c = 0; c < numClusters; ++c)
	{
		double currentdist = 0;

		for (size_t i = 0; i < dimension; i++) // p.getSize() or dimension = N
			currentdist += pow((p[i] - centroids[offset + (c*dimension) + i]), 2);

		if (currentdist < dist || dist == std::numeric_limits<double>::max())
		{
			dist = currentdist;
			indexCentroid = c;
		}
	}
	return indexCentroid;
}

/*
	Average of all points from a given cluster
	@param centroidIndex: the index of the centroid
	@param clusters: vector with the closest centroid per point
	@param clusterOffset: the offset in the cluster vector to seperate repetitions
	@param allPoints: all the points from the input file
	@return the average point from the cluster
*/
point average_of_points_with_cluster(const size_t centroidIndex, const std::vector<int> &clusters, const size_t clusterOffset, std::vector<double> &allPoints, const int dimension)
{
	point avgPoint;
	size_t numberOfPoints = 0;
	for (size_t i = 0; i < allPoints.size()/dimension; i++)
	{
		if (clusters[clusterOffset + i] == centroidIndex)
		{
			point p;
			for (size_t j = 0; j < dimension; j++)
				p.addDataPoint(allPoints[i*dimension+j]);
			avgPoint.add(p);
			numberOfPoints++;
		}
	}
	avgPoint.divide(numberOfPoints);
	return avgPoint;
}

/*
	Writes the clusters to the debug file
	@param numpoints: total number of points
	@param cluster: vector with the closest centroid per point
	@param clusterDebugFileName: the file name of the debug file
*/
void writeClusterToDebugFile(std::vector<double> &cluster, std::string &clusterDebugFileName, const int numpoints){
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
void writeCentroidToDebugFile(std::vector<double> &centroid, std::string &centroidDebugFileName, const int dimension){
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
			   int &bestClusterOffset,
			   std::vector<double> &centroids,
			   const size_t centroidOffset,
			   std::vector<int> &clusters,
			   const size_t clusterOffset,
			   std::vector<double> &allPoints,
			   const size_t numPoints,
			   const int numClusters,
			   bool debugCentroids,
			   bool debugClusters,
			   std::string &centroidDebugFile,
			   std::string &clustersDebugFile,
			   const int dimension
			   )
{

	bool changed = true;
	int steps = 0;
	
	// std::vector<double> debugCluster;
	// std::vector<double> debugCentroid;
	while (changed)
	{
		steps++;
		changed = false;
		double distanceSquaredSum = 0.0;
		
		for (int p = 0; p < numPoints; ++p)
		{
			double dist = std::numeric_limits<double>::max();
			const int newCluster = find_closest_centroid_index_and_distance(dist, &allPoints[p*dimension], centroids, numClusters, centroidOffset, dimension);
			distanceSquaredSum += dist;

			if (newCluster != clusters[clusterOffset + p])
			{
				clusters[clusterOffset + p] = newCluster;
				// printf("rank: %d, index: %d\n", mpi_rank, clusterOffset + p);
				changed = true;
			}
		}

		// printf("proces %d in with offset %f,%f\n", mpi_rank, centroids[centroidOffset],centroids[centroidOffset+1]);
	

		// printf("proces %d in function with sum %d\n", mpi_rank, distanceSquaredSum);

		// if(debugClusters)
		// 	debugCluster.insert(debugCluster.end(), &clusters[0], &clusters[numPoints]);	
		// if(debugCentroids){
		// 	for (size_t j = 0; j < numClusters; j++){
		// 		if(debugCentroids){
		// 			for (size_t i = 0; i < allPoints[0].getSize(); i++)
		// 				debugCentroid.push_back(centroids[j].getDataPoint(i));
		// 		}
		// 	}
		// }

		if (changed)
		{
			for (size_t j = 0; j < numClusters; ++j)
				for (size_t k = 0; k < dimension; ++k)
					centroids[centroidOffset + (j*dimension) + k] = average_of_points_with_cluster(j, clusters, clusterOffset, allPoints,dimension).getDataPoint(k);	
		}

		if (distanceSquaredSum < bestDistSquaredSum)
		{
			bestClusterOffset = clusterOffset;
			bestDistSquaredSum = distanceSquaredSum;
		}
	}
	// if(debugClusters)
	// 	writeClusterToDebugFile(debugCluster, clustersDebugFile, numPoints);
	// if(debugCentroids)
	// 	writeCentroidToDebugFile(debugCentroid, centroidDebugFile, allPoints[0].getSize());
	return steps;
}

int kmeans(Rng &rng,
		   const std::string &inputFile,
		   const std::string &outputFileName,
		   int numClusters,
		   int repetitions,
		   int numBlocks,
		   int numThreads,
		   std::string &centroidDebugFileName,
		   std::string &clusterDebugFileName)
{

	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	FileCSVWriter csvOutputFile;
	if (mpi_rank == mpi_rootRank){
		csvOutputFile.open(outputFileName);
		if (!csvOutputFile.is_open())
		{
			std::cerr << "Unable to open output file " << outputFileName << std::endl;
			return -1;
		}
	}

	static size_t numPoints = 0; // Will reset in the readData function anyways = rows
	static size_t dimension = 0; // Will reset in the readData function anyways = cols
	std::vector<double> allPoints;
	std::ifstream infile(inputFile);
	if (mpi_rank == mpi_rootRank)
		readData(infile, allPoints, numPoints, dimension);
	
	// THEN start timing! (don't want to also time the creation of our big variables, make it upfront and CONSTANT overhead)
	// This is a basic timer from std::chrono ; feel free to use the appropriate timer for
	// each of the technologies, e.g. OpenMP has omp_get_wtime()
	double t1 = MPI_Wtime();

	MPI_Bcast(&numPoints, 1, MPI_INT,0,MPI_COMM_WORLD); // broadcast number of points (numPoints)
	MPI_Bcast(&dimension, 1, MPI_INT,0,MPI_COMM_WORLD); // broadcast K (dimension)

	// initialize BIG variabels
	int bestClusterLocalOffset=0; //array
	std::vector<int> bestClusterOffset(repetitions,0); //array
	std::vector<int> clusters ((int)numPoints * repetitions,-1);
	/**
	 * with amount of -1 in the matrix = number of points * repetitions
	 * abstract example: [-1, -1, | -1, -1, | -1, -1]
	 * the | (abstract, not actually in the vector) defines a new repetition,
	 * in this case there are only 2 points in the entire csv file (2 rows),
	 * and 3 repetitions in total
	 */

	std::vector<double> centroids(numClusters * repetitions * dimension);
	/**
	 * centroids * repetitions
	 * abstract example: [p1, p2, p3 | p1, p2, p3]
	 * here are 3 centroids for 3 repetitions made
	*/

	if(mpi_rank == mpi_rootRank)
		choose_centroids_at_random(numClusters, dimension, rng, centroids, repetitions, allPoints);

	if(mpi_rank != mpi_rootRank)
		allPoints.resize(numPoints*dimension);
	MPI_Bcast(allPoints.data(), (int)allPoints.size(),MPI_DOUBLE,0,MPI_COMM_WORLD); //broadcast allpoints 

	std::vector<double> bestDistSquaredSum(repetitions,std::numeric_limits<double>::max()); // can only get better
	double bestDistSquaredLocalSum = std::numeric_limits<double>::max(); // can only get better
	std::vector<int> stepsPerRepetition(repetitions,0);			// to save the number of steps each rep needed
	
	// Scatterv - centroids
	std::vector<int> counts_centroids(mpi_size,0);
	std::vector<int> displs_centroids(mpi_size,0);
	for (size_t i = 0; i < repetitions; i++)// number of reps per mpi proces
		counts_centroids[i % mpi_size] += numClusters * dimension;
	for (size_t i=1; i<mpi_size; i++)
		displs_centroids[i] = displs_centroids[i-1] + counts_centroids[i-1];

	// Gatherv - clusters + Determine cluster location
	std::vector<int> counts_clusters(mpi_size,0);
	std::vector<int> displs_clusters(mpi_size,0);
	for (size_t i = 0; i < repetitions; i++) // number of clusters per mpi proces
		counts_clusters[i % mpi_size] += numPoints;
	for (size_t i=1; i<mpi_size; i++)
		displs_clusters[i] = displs_clusters[i-1] + counts_clusters[i-1];
	
	// Gatherv - steps
	std::vector<int> counts_steps(mpi_size,0);
	std::vector<int> displs_steps(mpi_size,0);
	for (size_t i = 0; i < repetitions; i++) // number of clusters per mpi proces
		counts_steps[i % mpi_size] += 1;
	for (size_t i=1; i<mpi_size; i++)
		displs_steps[i] = displs_steps[i-1] + counts_steps[i-1];
	
	// Do the k-means routine a number of times, each time starting from
	// different random centroids (use Rng::pickRandomIndices), and keep
	// the best result of these repetitions.
	MPI_Scatterv(centroids.data(),counts_centroids.data(),displs_centroids.data(),MPI_DOUBLE,centroids.data(),counts_centroids[mpi_rank],MPI_DOUBLE,mpi_rootRank,MPI_COMM_WORLD);

	int numberOfRepsPerRank = counts_centroids[mpi_rank]/(numClusters*dimension);
	int recv_steps[numberOfRepsPerRank];
	int cluster_location = 0;
	for (size_t i = 0; i < mpi_rank; i++)
		cluster_location += counts_clusters[i];

	// for (int r = mpi_rank*partition; r < mpi_rank*partition+partition; r++)
	for (int r = 0; r < numberOfRepsPerRank; r++){
		recv_steps[r] = kmeansReps(bestDistSquaredLocalSum, bestClusterLocalOffset, centroids, numClusters*dimension*r, clusters, numPoints*r, allPoints, numPoints, numClusters, false, false, centroidDebugFileName, clusterDebugFileName, dimension);
	}
	// printf("rank: %d, bestclusterOffset: %d, sum: %f\n", mpi_rank, bestClusterLocalOffset, bestDistSquaredLocalSum);
	bestClusterLocalOffset += cluster_location;
	MPI_Gatherv(clusters.data(), numPoints*numberOfRepsPerRank, MPI_INT, clusters.data(), counts_clusters.data(), displs_clusters.data(), MPI_INT, mpi_rootRank,
            MPI_COMM_WORLD);
	MPI_Gatherv(recv_steps, numberOfRepsPerRank, MPI_INT, stepsPerRepetition.data(), counts_steps.data(), displs_steps.data(), MPI_INT, mpi_rootRank, 
            MPI_COMM_WORLD);
	MPI_Gather(&bestDistSquaredLocalSum, 1, MPI_DOUBLE, bestDistSquaredSum.data(),1, MPI_DOUBLE, mpi_rootRank,
				MPI_COMM_WORLD);
	MPI_Gather(&bestClusterLocalOffset, 1, MPI_INT, bestClusterOffset.data(),1, MPI_INT, mpi_rootRank,
				MPI_COMM_WORLD);

	double t2 = MPI_Wtime();

	if (mpi_rank == mpi_rootRank){
		bestDistSquaredLocalSum = *std::min_element(bestDistSquaredSum.begin(),bestDistSquaredSum.end());
		std::vector<double>::iterator it = std::min_element(bestDistSquaredSum.begin(), bestDistSquaredSum.end());
		int smallestValue_index = std::distance(bestDistSquaredSum.begin(), it);
		bestClusterLocalOffset = bestClusterOffset[smallestValue_index];
		
		// Some example output, of course you can log your timing data anyway you like.
		std::cout << "# Type,blocks,threads,MPI processes,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
		std::cout << "MPI," << numBlocks << "," << numThreads << "," << mpi_size << "," << inputFile << ","
				<< rng.getUsedSeed() << "," << numClusters << ","
				<< repetitions << "," << bestDistSquaredLocalSum << "," << /*timer.durationNanoSeconds() / 1e9*/ t2-t1
				<< std::endl;

		// Write the number of steps per repetition, kind of a signature of the work involved
		csvOutputFile.write(stepsPerRepetition, "# Steps: ");
		// Write best clusters to csvOutputFile, something like

		// Get best cluster from repetitions -> example output shows the first cluster, we use the best cluster for output
		std::vector<int> bestCluster(&clusters[bestClusterLocalOffset], &clusters[bestClusterLocalOffset + numPoints]);
		csvOutputFile.write(bestCluster);
	}

	MPI_Finalize();
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
	MPI_Init(&argc, &argv);
	std::vector<std::string> args;
	for (int i = 1; i < argc; i++)
		args.push_back(argv[i]);

	return mainCxx(args);
}
