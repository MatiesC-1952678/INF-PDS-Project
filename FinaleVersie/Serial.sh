#!/bin/bash -l
#PBS -l nodes=1:ppn=36
#PBS -l walltime=0:30:00
#PBS -A llp_h_pds
#PBS -l qos=debugging
# module load GCC/8.3.0
# #module list

# cd $PBS_O_WORKDIR

inputDir="100000x5.csv"
k=10
rep=10

for ((i=1; i<=15; i++))
do
    ./kmeans_serial --input $inputDir  --output serial.csv --repetitions $rep --k $k --seed 1952226
done

for ((i=1; i<=15; i++))
do
    ./kmeans --input $inputDir --threads 1  --output openmp1.csv --repetitions $rep --k $k --seed 1952226
done
