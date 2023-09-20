#!/bin/bash
source ~/.bash_functions

mpic++ /home/steven/Documents/PostDoc/DW/mpi_evolution.cpp -o /home/steven/Documents/PostDoc/DW/Executables/mpi_evolution -O3
if [ $? -ne 0 ]
then
	echo "Compile failed"
else
	mpirun --use-hwthread-cpus Executables/mpi_evolution
	#mpirun --use-hwthread-cpus Executables/moore_mpi_evolution
	#mpirun Executables/mpi_evolution
fi
