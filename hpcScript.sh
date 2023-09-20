#!/bin/bash --login
#$ -cwd
#$ -pe hpc.pe 256
#$ -P hpc-rb-topo

mpirun -n $NSLOTS Executables/mpi_evolution$TAG






