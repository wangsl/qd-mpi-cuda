

/* $Id$ */

#include <iostream>
#include <cstring>
#include <cmath>
#include <mex.h>

#include "evolutionCUDA.h"
#include "evolutionUtils.h"

extern "C" int FORT(myisnan)(const double &x)
{
  return isnan(x);
}

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  const int np = std::cout.precision();
  std::cout.precision(15);
  
  std::cout << " Quantum Dynamics Time evolotion with CUDA and MPI" << std::endl;

  insist(nrhs == 1);

  mxArray *mxPtr = 0;
  
  mxPtr = mxGetField(prhs[0], 0, "r1");
  insist(mxPtr);
  RadialCoordinate r1(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "r2");
  insist(mxPtr);
  RadialCoordinate r2(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "theta");
  insist(mxPtr);
  AngleCoordinate theta(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "pot");
  insist(mxPtr);
  MatlabArray<double> pot(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "psi");
  insist(mxPtr);
  MatlabArray<Complex> psi(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "time");
  insist(mxPtr);
  EvolutionTime time(mxPtr);
  
  mxPtr = mxGetField(prhs[0], 0, "options");
  insist(mxPtr);
  Options options(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "dump1");
  insist(mxPtr);
  DumpFunction dump1(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "dump2");
  insist(mxPtr);
  DumpFunction dump2(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "CRP");
  insist(mxPtr);
  CummulativeReactionProbabilities CRP(mxPtr);

  mxPtr = mxGetField(prhs[0], 0, "OmegaStates");
  insist(mxPtr);
  OmegaStates omegas(mxPtr);

  //void _mpi_test_1();
  //_mpi_test_1();

  for(int j = 0; j < omegas.lmax; j++) 
    setup_coriolis_matrix(omegas.J, omegas.parity, j);

#if 0
  EvolutionCUDA evolCUDA(pot.data, r1, r2, theta, omegas, time);
  cudaUtils::gpu_memory_usage();
  evolCUDA.test();
#endif
 
  std::cout.flush();
  std::cout.precision(np);
}
