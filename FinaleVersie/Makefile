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

# clean:
# 	rm -f kmeans rng.o main_startcode.o

# kmeans: main_startcode.o rng.o
# 	$(MPICXX) -o kmeans main_startcode.o rng.o

# main_startcode.o: main_startcode.cpp
# 	$(MPICXX) -c main_startcode.cpp

# rng.o: rng.cpp
# 	$(CXX) -c rng.cpp -o rng.o

export OMPI_CXX=g++-12

clean:
	rm -f kmeans

kmeans: main_startcode.cpp rng.cpp
	$(MPICXX) -o kmeans ./main_startcode.cpp ./rng.cpp

run:
	mpirun -np 8 kmeans  --input /Users/yarne/Documents/datasets/mouse_500x2.csv --output /Users/yarne/Documents/MPI/FinaleVersie/output/mouse.csv --repetitions 16 --k 3 --seed 1952226

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