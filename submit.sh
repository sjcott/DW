#!/bin/bash

module load mpi/gcc/openmpi/4.1.0
module load compilers/gcc/9.3.0

# Put a tag on the executable so that if a job sits in the queue, the correct
# one is run
TAG=_2048_m3em6

################# MPI evolution compile and submit if successful ##################################

mpic++ ~/scratch/DW/mpi_evolution.cpp -o ~/scratch/DW/Executables/mpi_evolution$TAG -O3
if [ $? -ne 0 ]
then
	echo "Compile failed"
else
	qsub hpcScript.sh
fi





