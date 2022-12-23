# -std=c++14: we're limiting ourselves to c++14, since that's what the 
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
#FLAGS=-O3 -DNDEBUG -std=c++14

# Settings for a debug build
FLAGS_COLLAB=-std=c++14 -gencode arch=compute_75,code=sm_75 
FLAGS_VSC=-std=c++14 -gencode arch=compute_60,code=sm_60 
MAIN_CALC_CENTROID_GPU=main_centroidGPU.cu
MAIN_CALC_CENTROID_CPU=main_centroidCPU.cu

NVCC=nvcc

all: kmeans

clean:
	rm -f kmeans

kmeans: main_startcode.cpp rng.cpp
	$(NVCC) $(FLAGS_COLLAB)  -o kmeans $(MAIN_CALC_CENTROID_GPU) rng.cpp
