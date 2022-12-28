# -std=c++14: we're limiting ourselves to c++14, since that's what the 
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
#FLAGS=-O3 -DNDEBUG #-std=c++14

# Settings for a debug build
FLAGS= -std=c++14

CXX=g++-12
MPICXX=mpicxx

all: kmeans

export OMPI_CXX=g++-12

clean:
	rm -f kmeans

kmeans: main_startcode.cpp rng.cpp
	$(MPICXX) -o kmeans_mpi ./main_startcode.cpp ./rng.cpp

run:
	mpirun -np 3 kmeans_mpi --input /Users/yarne/Documents/datasets/mouse_500x2.csv --output /Users/yarne/Documents/MPI/FinaleVersie/output/mouse_500x2.csv --repetitions 4 --k 6 --seed 1952226

# # Settings for a debug build
# FLAGS=-g -std=c++14

# CXX = g++-12

# all: kmeans

# clean:
# 	rm -f kmeans

# kmeans: main_startcode.cpp rng.cpp
# 	$(CXX) $(FLAGS) -o kmeans main_startcode.cpp rng.cpp

# run: 
# 	./kmeans  --input /Users/yarne/Documents/datasets/mouse_500x2.csv --output /Users/yarne/Documents/MPI/FinaleVersie/output/mouse.csv --repetitions 10 --k 3 --seed 1952226