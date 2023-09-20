#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <vector>
#include <mpi.h>
#include <random>

using namespace std;

const double pi = 4.0*atan(1.0);

// Never adjusted but useful to define
const int nts = 2; // Number of time steps saved in data arrays

const long long int nx = 2048;
const long long int ny = 2048;
const long long int nz = 2048;
const long long int nPos = nx*ny*nz;
// nt required for sim to end at light crossing time is nx*dx/(2*dt). nt_max from resolving defect (in standard rad dom. expanding universe) is ~ 1/(dt*dx). Therefore, ideally pick dx = sqrt(2/nx).
const int nt = 5*nx/2; 
const double dx = sqrt(2.0/nx);
const double dy = sqrt(2.0/ny);
const double dz = sqrt(2.0/nz);
const double dt = (dx+dy+dz)/(15); // Averages the other grid spacings and divides by 5

const double lambda = 1;
const double eta = 1;
const double m1 = 0;
const int ntm1 = 0; // Switch on m1 at this timestep
const double m3 = 1e-4;
const int ntm3 = 0; // Switch on m3 at this timestep

const int damped_nt = round(25*sqrt(nx/2.0)); // Number of time steps for which damping is imposed. Useful for random initial conditions
const double dampFac = 1.0; // magnitude of damping term, unclear how strong to make this
const int ntHeld = 0; // Hold fields fixed (but effectively continue expansion) for this number of timesteps. Attempting to get the network into the scaling regime. Not sure how useful this is...
const bool expandDamp = false; // If true then the universe expands during the damping regime.

// Below has not been implemented for gauge fields yet - so only works with global strings.
// Set alpha to zero to recover a non-expanding univlerse. Note that non-zero is not standard expansion but PRS algorithm.

const double alpha = 2; // Factor multiplying hubble damping term for use in PRS algorithm. alpha = #dims has been claimed to give similar dynamics without changing string width.
                        // alpha = #dims - 1 is the usual factor
const double beta = 2; // scale factor^beta is the factor that multiplies the potential contribution to the EoMs. Standard is 2, PRS is 0.
const double scaling = 1; // Power law scaling of the scale factor wrt tau. Using conformal time so rad dom is gamma=1 while matter dom is gamma=2. gamma=0 returns a static universe

const bool makeGif = false; // Outputs data to make a gif of the isosurfaces. Not implemented in this version yet.
const int saveFreq = 5;
const int countRate = 10;
const string outTag = "Z2_m3em4";

// How are the initial conditions generated? Below are all parameters used for generating (or loading) the initial conditions
const string ic_type = "random"; // Current options are data, stationary data and random
const int seed = 42;

// Below has been removed and code now assumes periodic boundaries. Shouldn't be too tricky to add it back in if neccessary

//const string xyBC = "periodic"; // Allows for "neumann" (covariant derivatives set to zero), "absorbing", "periodic" or fixed (any other string will choose this option) boundary conditions.
//const string zBC = "periodic"; // Allows for "neumann" (covariant derivatives set to zero), "periodic" or fixed (any other string will choose this option) boundary conditions.

const bool calcEnergy = true;
const bool wallDetect = true;
const bool finalOut = false;

int main(int argc, char ** argv){

	// Initialize MPI

    // Init MPI
    MPI_Init( &argc, &argv);

    // Get the rank and size
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    long long int chunk = nPos/size;
    long long int chunkRem = nPos - size*chunk;

    long long int coreSize;
	if(rank>=chunkRem){ coreSize = chunk; }
	else{ coreSize = chunk+1; }

	// Calculate the position of the start of the chunk in the full array
	long long int coreStart, coreEnd;
    if(rank < chunkRem){ coreStart = rank*(chunk+1); coreEnd = (rank+1)*(chunk+1); }
    else{ coreStart = rank*chunk+chunkRem; coreEnd = (rank+1)*chunk+chunkRem; }

    // Old method - just making haloes large enough to be sure it would fit.
    // int haloSize = 2*nz*ny;
    // int totSize = coreSize + 2*haloSize;

    // Calculate the halo sizes (all data up to the previous x row at start of chunk and all data up to the next x row at the end of the chunk)
    long long int frontHaloSize, backHaloSize, nbrFrontHaloSize, nbrBackHaloSize, remFront, remBack;
    remFront = coreStart%(ny*nz);
    remBack = coreEnd%(ny*nz);
    if(remFront==0){ // Smallest possible halo size

    	frontHaloSize = ny*nz;
    	nbrBackHaloSize = ny*nz;

    } else{

    	// The two sum to 3*ny*nz rather than 2*ny*nz. This is inefficient and should be avoided if possible.

    	frontHaloSize = ny*nz + remFront;
    	nbrBackHaloSize = 2*ny*nz - remFront;

    }

    if(remBack==0){

    	backHaloSize = ny*nz;
    	nbrFrontHaloSize = ny*nz;

    } else{

    	backHaloSize = 2*ny*nz - remBack;
    	nbrFrontHaloSize = ny*nz + remBack;

    }

    // Size the array needs to be to hold the core and the two halos.
    long long int totSize = frontHaloSize + coreSize + backHaloSize;

    // Calculate the position of the start of the local array (including the haloes) in the full array. This quantity wraps around (i.e -ve numbers mean the other side of array)
    long long int dataStart = coreStart-frontHaloSize;
    long long int dataEnd = coreEnd+backHaloSize;

    // Warnings

    if(rank==0){

    	if(size==1){ cout << "Warning: Only one processor being used. This code is not designed for only one processor and may not work." << endl; }
    	if(chunk<ny*nz){ cout << "Warning: Chunk size is less than the minimum halo size (i.e chunk neighbour data). Code currently assumes this is not the case so it probably won't work." << endl; }

    }

	vector<double> phi(2*totSize, 0.0);
	double phixx,phiyy,phizz,localEnergy,localNDW,localADW_simple,localADW_full,damp,phit,phiMagSqr,localSeed,phitt,phix,phiy,phiz,x0,y0,z0;
	long long int i,j,k,TimeStep,gifStringPosFrame,tNow,tPast,counter,comp,imx,ipx,imy,ipy,imz,ipz,ipxmy,ipxmz,imxpy,ipymz,imxpz,imypz;

	struct timeval start, end;
	if(rank==0){ gettimeofday(&start, NULL); }

    string file_path = __FILE__;
    string dir_path = file_path.substr(0,file_path.find_last_of('/'));
    stringstream ss;

    MPI_Barrier(MPI_COMM_WORLD);

    // string input;
    // if(rank==0){

    // 	cout << "Enter a tag for output files: " << flush;
    // 	cin >> input;

    // }

    MPI_Barrier(MPI_COMM_WORLD); // Allows all other processes to start once user input has been received.

    string icPath = dir_path + "/Data/ic.txt";
    string finalFieldPath = dir_path + "/Data/finalField.txt";
    string valsPerLoopPath = dir_path + "/Data/valsPerLoop_" + outTag + "_nx" + to_string(nx) + "_dnt" + to_string(damped_nt) + "_seed" + to_string(seed) + ".txt";

    ifstream ic (icPath.c_str());
    ofstream finalField (finalFieldPath.c_str());
    ofstream valsPerLoop (valsPerLoopPath.c_str());

    // Index values (not neccessarily on grid and hence not integers) of the zero coordinate.
    x0 = 0.5*(nx-1);
    y0 = 0.5*(ny-1);
    z0 = 0.5*(nz-1);

    double wasteData;

    if(ic_type=="stationary data"){

    	for(i=0;i<nPos;i++){

    		// Only assign it to the local array if this point belongs to the core or halo. Otherwise just waste it. Use modulus operator to deal with periodicity
    		// Uses index calculation totSize*(nTimeSteps*component + TimeStep) + pos

    		if(dataStart<0){ // Need to deal with periodicity at the start of the data array

    			if(i>=(dataStart+nPos)%nPos){ // Front halo

    				long long int arrayPos = i-(dataStart+nPos)%nPos;

    				ic >> phi[arrayPos];

    				// Second time step is equal to the first

    				phi[totSize+arrayPos] = phi[arrayPos];

    			} else if(i<dataEnd){ // The rest of the data

    				long long int arrayPos = i+frontHaloSize; // Shift across to account for the front halo at the start

    				ic >> phi[arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];

    			} else{ ic >> wasteData; } // Don't need these so waste them into an unused variable 


    		} else if(dataEnd>nPos){ // Need to deal with periodicity at the end of the data array

    			if(i>=dataStart){ // All of the array except for the back halo

    				long long int arrayPos = i-dataStart;

					ic >> phi[arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];; 				

    			} else if(i<dataEnd%nPos){ // The back halo

    				long long int arrayPos = i+coreSize+frontHaloSize;

    				ic >> phi[arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];

    			} else{ ic >> wasteData; }

    		} else{ // In the middle of the array so don't need to deal with periodicity

    			if(i>=dataStart and i<dataEnd){

    				long long int arrayPos = i-dataStart;

    				ic >> phi[arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];

    			} else{ ic >> wasteData; }

    		}

    	}

    } else if(ic_type=="data"){

    	for(TimeStep=0;TimeStep<2;TimeStep++){
    		for(i=0;i<nPos;i++){

    			if(dataStart<0){ // Need to deal with periodicity at the start of the data array

	    			if(i>=(dataStart+nPos)%nPos){ // Front halo

	    				long long int arrayPos = i-(dataStart+nPos)%nPos;

	    				ic >> phi[totSize*TimeStep+arrayPos];

	    			} else if(i<dataEnd){ // The rest of the data

	    				long long int arrayPos = i+frontHaloSize; // Shift across to account for the front halo at the start

	    				ic >> phi[totSize*TimeStep+arrayPos];

	    			} else{ ic >> wasteData; } // Don't need these so waste them into an unused variable 


	    		} else if(dataEnd>nPos){ // Need to deal with periodicity at the end of the data array

	    			if(i>=dataStart){ // All of the array except for the back halo

	    				long long int arrayPos = i-dataStart;

						ic >> phi[totSize*TimeStep+arrayPos];				

	    			} else if(i<dataEnd%nPos){ // The back halo

	    				long long int arrayPos = i+coreSize+frontHaloSize;

	    				ic >> phi[totSize*TimeStep+arrayPos];

	    			} else{ ic >> wasteData; }

	    		} else{ // In the middle of the array so don't need to deal with periodicity

	    			if(i>=dataStart and i<dataEnd){

	    				long long int arrayPos = i-dataStart;

	    				ic >> phi[totSize*TimeStep+arrayPos];

	    			} else{ ic >> wasteData; }

	    		}

    		}
    	}

    } else if(ic_type=="random"){

        // Currently assumes the vacuum is phi=+/-eta which won't be the case when the "soft breaking" terms are non-zero.

    	// Use the seed to generate the data
		mt19937 generator (seed);
        uniform_real_distribution<double> distribution (-1.0, 1.0); // Uniform distribution for the phase of the strings
        double phiAssign;

        // Skip the random numbers ahead to the appropriate point.
        for(i=0;i<coreStart;i++){ phiAssign = distribution(generator); }



        for(i=frontHaloSize;i<coreSize+frontHaloSize;i++){

        	phiAssign = distribution(generator);

        	phi[i] = eta*phiAssign;

        	// Set next timestep as equal to the first
        	phi[totSize+i] = phi[i];

        }

        //cout << "Rank " << rank << "has phi[haloSize] = " << phi[haloSize] << ", and the next random number would be " << distribution(generator) << endl;

        // Now that the core data has been generated, need to communicate the haloes between processes. 


    	MPI_Sendrecv(&phi[frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,0, // Send this
    				 &phi[coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // Receive this

    	MPI_Sendrecv(&phi[coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,0,
    				 &phi[0],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    	MPI_Sendrecv(&phi[totSize+frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,1,
    				 &phi[totSize+coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    	MPI_Sendrecv(&phi[totSize+coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,1,
    				 &phi[totSize],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 	

    }

    gettimeofday(&end,NULL);

    if(rank==0){ cout << "Initial data loaded/generated in: " << end.tv_sec - start.tv_sec << "s" << endl; }

    for(TimeStep=0;TimeStep<nt;TimeStep++){

        //double time = 1 + TimeStep*dt; // Conformal time, starting at eta = 1.
        double fric, tau;

        if(TimeStep>counter and rank==0){

            cout << "\rTimestep " << TimeStep-1 << " completed." << flush;

            counter += countRate;

        }

        if(expandDamp){ tau = 1 + (ntHeld+TimeStep)*dt; }
        else{ tau = 1 + (ntHeld+TimeStep-damped_nt)*dt; }

        // Is damping switched on or not?
        if(TimeStep<damped_nt){

        	if(expandDamp){ fric = dampFac + alpha*scaling/tau; } // denominator is conformal time
        	else{ fric = dampFac; }

        } else{

        	if(expandDamp){ fric = alpha*scaling/tau; } // Time needs to have moved along during the damped phase
			else{ fric = alpha*scaling/tau; } // Time was not progressing during the damped phase

        }

        tNow = (TimeStep+1)%2;
        tPast = TimeStep%2;


        // Calculate time derivatives using EoMs

        localEnergy = 0;
        localNDW = 0;
        localADW_simple = 0;
        localADW_full = 0;

        for(i=frontHaloSize;i<coreSize+frontHaloSize;i++){ // Now evolve the core data

        	// No need to worry about periodicity with the x neighbours because halo is designed to contain them

        	imx = i-ny*nz;
        	ipx = i+ny*nz;

        	// Need to account for the periodicity of the space for the other two directions

        	// Convert to global position in array to do modulo arithmetic. The second to last term gives ny*nz*floor((i+dataStart)/(ny*nz)). The last term converts back to the position in the local array 

        	imy = (i+dataStart-nz+ny*nz)%(ny*nz) + ( (i+dataStart)/(ny*nz) )*ny*nz - dataStart; 
        	ipy = (i+dataStart+nz)%(ny*nz) + ( (i+dataStart)/(ny*nz) )*ny*nz - dataStart;

        	imz = (i+dataStart-1+nz)%nz + ( (i+dataStart)/nz )*nz - dataStart;
        	ipz = (i+dataStart+1)%nz + ( (i+dataStart)/nz )*nz - dataStart;

        	// Additionally needed for wilson loop calculations. Avoid using x shifted points first as this makes the calculations more complicated and some of these points aren't in the correct positions

        	ipxmy = imy+ny*nz;
        	ipxmz = imz+ny*nz;
        	imxpy = ipy-ny*nz;
        	ipymz = (ipy+dataStart-1+nz)%nz + ( (ipy+dataStart)/nz )*nz - dataStart;
        	imxpz = ipz-ny*nz;
        	imypz = (imy+dataStart+1)%nz + ( (imy+dataStart)/nz )*nz - dataStart;


	        phiMagSqr = pow(phi[totSize*tNow+i],2);

             // 2nd order spatial derivatives calculated with 2nd order finite difference

            phixx = ( phi[totSize*tNow+ipx] - 2*phi[totSize*tNow+i] + phi[totSize*tNow+imx] )/(dx*dx);

            phiyy = ( phi[totSize*tNow+ipy] - 2*phi[totSize*tNow+i]	+ phi[totSize*tNow+imy] )/(dy*dy);

            phizz = ( phi[totSize*tNow+ipz] - 2*phi[totSize*tNow+i]	+ phi[totSize*tNow+imz] )/(dz*dz);


            phit = ( phi[totSize*tNow+i] - phi[totSize*tPast+i] )/dt;

            // Calculate the second order time derivative and update the field
            phitt = phixx + phiyy + phizz - pow( pow(tau,scaling), beta )*( 0.5*lambda*(phiMagSqr - pow(eta,2))*phi[totSize*tNow+i] ) - fric*phit;
            if(TimeStep>=ntm1){ phitt -= pow( pow(tau,scaling), beta )*0.5*m1; }
            if(TimeStep>=ntm3){ phitt -= pow( pow(tau,scaling), beta )*1.5*m3*phiMagSqr; }
            phi[totSize*tPast+i] = 2*phi[totSize*tNow+i] - phi[totSize*tPast+i] + dt*dt*phitt;


            // Calculate the energy contained in this process's domain
            if(calcEnergy){

                phix = ( phi[totSize*tNow+i] - phi[totSize*tNow+imx] )/dx;
                phiy = ( phi[totSize*tNow+i] - phi[totSize*tNow+imy] )/dy;
                phiz = ( phi[totSize*tNow+i] - phi[totSize*tNow+imz] )/dz;

                localEnergy += ( pow(phit,2) + pow(phix,2) + pow(phiy,2) + pow(phiz,2) + 0.25*lambda*pow(phiMagSqr-pow(eta,2),2) )*dx*dy*dz;
                if(TimeStep>=ntm1){ localEnergy += m1*phi[totSize*tNow+i]*dx*dy*dz; }
                if(TimeStep>=ntm3){ localEnergy += m3*phiMagSqr*phi[totSize*tNow+i]*dx*dy*dz; }

            }

            // If the sign of phi flips between any two neighbours, consider that as a wall detection. Sum this up. Only look at forward neighbours so I'm not double counting.
            if(wallDetect){

                // x neighbour
                if(phi[totSize*tNow+i]*phi[totSize*tNow+ipx]<0){ 

                    localNDW += 1;
                    localADW_simple += 2.0*dy*dz/3.0;

                    phix = ( phi[totSize*tNow+i] - phi[totSize*tNow+imx] )/dx;
                    phiy = ( phi[totSize*tNow+i] - phi[totSize*tNow+imy] )/dy;
                    phiz = ( phi[totSize*tNow+i] - phi[totSize*tNow+imz] )/dz;
                    localADW_full += dy*dz*sqrt(pow(phix,2)+pow(phiy,2)+pow(phiz,2))/( abs(phix) + abs(phiy) + abs(phiz) ); 

                }

                // y neighbour
                if(phi[totSize*tNow+i]*phi[totSize*tNow+ipy]<0){ 

                    localNDW += 1;
                    localADW_simple += 2.0*dx*dz/3.0;

                    phix = ( phi[totSize*tNow+i] - phi[totSize*tNow+imx] )/dx;
                    phiy = ( phi[totSize*tNow+i] - phi[totSize*tNow+imy] )/dy;
                    phiz = ( phi[totSize*tNow+i] - phi[totSize*tNow+imz] )/dz;
                    localADW_full += dx*dz*sqrt(pow(phix,2)+pow(phiy,2)+pow(phiz,2))/( abs(phix) + abs(phiy) + abs(phiz) );

                }

                // z neighbour
                if(phi[totSize*tNow+i]*phi[totSize*tNow+ipz]<0){ 

                    localNDW += 1;
                    localADW_simple += 2.0*dx*dy/3.0;

                    phix = ( phi[totSize*tNow+i] - phi[totSize*tNow+imx] )/dx;
                    phiy = ( phi[totSize*tNow+i] - phi[totSize*tNow+imy] )/dy;
                    phiz = ( phi[totSize*tNow+i] - phi[totSize*tNow+imz] )/dz;
                    localADW_full += dx*dy*sqrt(pow(phix,2)+pow(phiy,2)+pow(phiz,2))/( abs(phix) + abs(phiy) + abs(phiz) );

                }

            }

        }

	    // If calculating the energy, add it all up and output to text
        if(calcEnergy){

            if(rank==0){

                double energy = localEnergy; // Initialise the energy as the energy in the domain of this process. Then add the energy in the regions of the other processes.

                for(i=1;i<size;i++){ MPI_Recv(&localEnergy,1,MPI_DOUBLE,i,5,MPI_COMM_WORLD,MPI_STATUS_IGNORE);  energy += localEnergy; }

                valsPerLoop << energy << " ";

            } else{ MPI_Send(&localEnergy,1,MPI_DOUBLE,0,5,MPI_COMM_WORLD); }

        }

        // Sum up the locally detected walls and output to text
        if(wallDetect){

            if(rank==0){

                double NDW = localNDW;
                double ADW_simple = localADW_simple;
                double ADW_full = localADW_full;

                for(i=1;i<size;i++){ 

                    MPI_Recv(&localNDW,1,MPI_DOUBLE,i,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
                    NDW += localNDW;

                    MPI_Recv(&localADW_simple,1,MPI_DOUBLE,i,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
                    ADW_simple += localADW_simple;

                    MPI_Recv(&localADW_full,1,MPI_DOUBLE,i,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
                    ADW_full += localADW_full;

                    }

                valsPerLoop << NDW << " " << ADW_simple << " " << ADW_full << " ";

            } else{ 

                MPI_Send(&localNDW,1,MPI_DOUBLE,0,6,MPI_COMM_WORLD);
                MPI_Send(&localADW_simple,1,MPI_DOUBLE,0,6,MPI_COMM_WORLD);
                MPI_Send(&localADW_full,1,MPI_DOUBLE,0,6,MPI_COMM_WORLD);

                }

        }

        if(rank==0){ valsPerLoop << endl; }

        // Update the core

        // Send sections of the core that are haloes for the other processes across to the relevant process. Then receive data for the halo of this process.

    	MPI_Sendrecv(&phi[totSize*tPast+frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,0, // Send this
    				 &phi[totSize*tPast+coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // Receive this

    	MPI_Sendrecv(&phi[totSize*tPast+coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,0,
    				 &phi[totSize*tPast],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        //Output the final fields. Possibly a less memory intensive way to do this is to let each node output one at a time by sending onward messages when they are done.

        if(finalOut and TimeStep==nt-1){

	        if(rank==0){

	        	vector<double> phiOut(nPos,0.0);
	        	for(i=0;i<coreSize;i++){ phiOut[i] = phi[frontHaloSize+i]; }

	        	for(i=1;i<size;i++){

	        		int localCoreStart;
	        		int localCoreSize;
	        		if(i<chunkRem){ localCoreStart = i*(chunk+1); localCoreSize = chunk+1; }
	        		else{ localCoreStart = i*chunk + chunkRem; localCoreSize = chunk; }

	        		MPI_Recv(&phiOut[localCoreStart],localCoreSize,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 

				}

	        	for(i=0;i<nPos;i++){ finalField << phiOut[i] << endl; }



	        } else{ MPI_Send(&phi[frontHaloSize],coreSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD); }

		}

    // Barrier before going to the next timestep. Not sure if strictly neccessary but I'm a paranoid man.

    MPI_Barrier(MPI_COMM_WORLD);

    }

    if(rank==0){

	    cout << "\rTimestep " << nt << " completed." << endl;

	    gettimeofday(&end,NULL);

	    cout << "Time taken: " << end.tv_sec - start.tv_sec << "s" << endl;

	}


    MPI_Finalize();


	return 0;

}
