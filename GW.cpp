// GW function file
// Performs FTs of energy-momemtum tensor to calculate the power specturm of gravitational waves emitted throughout the simulation.

#include <fftw3-mpi.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "GW.hpp"

void GW(phi){

	fftw_plan plan;
    double *Txx, *Tyy, *Tzz, *Txy, *Txz, *Tyz;
    fftw_complex *FTxx, *FTyy, *FTzz, *FTxy, *FTxz, *FTyz;
    fftw_mpi_init();

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    long long int chunk = nPos/size;
    long long int chunkRem = nPos - size*chunk;

    long long int coreSize;
	if(rank>=chunkRem){ coreSize = chunk; }
	else{ coreSize = chunk+1; }

	// Recalculate the shares for the padded real data
	chunk = (nx*ny*(nz/2+1))/size;
	chunkRem = (nx*ny*(nz/2+1)) - size*chunk;

	long long int fftSize;
	if(rank>=chunkRem){ fftSize = chunk; }
	else{ fftSize = chunk+1; }

	// Allocate the energy-momentum tensor components
	Txx = fftw_alloc_real(2*fftSize);
	Txy = fftw_alloc_real(2*fftSize);
	Txz = fftw_alloc_real(2*fftSize);
	Tyy = fftw_alloc_real(2*fftSize);
	Tyz = fftw_alloc_real(2*fftSize);
	Tzz = fftw_alloc_real(2*fftSize);

	// Allocate the fourier transforms of each component
    FTxx = fftw_alloc_complex(fftSize);
    FTxy = fftw_alloc_complex(fftSize);
    FTxz = fftw_alloc_complex(fftSize);
    FTyy = fftw_alloc_complex(fftSize);
    FTyz = fftw_alloc_complex(fftSize);
    FTzz = fftw_alloc_complex(fftSize);
    //std::cout << alloc_local << std::endl;

    /* create plan for out-of-place r2c DFT */
    plan = fftw_mpi_plan_dft_r2c_3d(L, M, N, rin, cout, MPI_COMM_WORLD,
                                    FFTW_MEASURE);

	
	
	return 0;

}