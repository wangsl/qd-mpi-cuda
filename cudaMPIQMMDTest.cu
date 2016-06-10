
#include <iostream>
#include <mpi.h>
#include <helper_cuda.h>
#include "matlabUtils.h"

#define _NTHREADS_ 512

inline int number_of_blocks(const int n_threads, const int n)
{ return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

static __global__ void test_pot(double *p, const int n)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    const double x = sin(p[index]);
}

void _cuda_mpi_test_for_quantum_dynamics_1(double *pot, const int n)
{
  int rank = -1;
  insist(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

  int n_ranks = 0;
  insist(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks) == MPI_SUCCESS);
 
  std::cout << " n = " << n << std::endl;
  
  double *pot_dev = 0;
  checkCudaErrors(cudaMalloc(&pot_dev, n*sizeof(double)));
  insist(pot_dev);
  checkCudaErrors(cudaMemcpy(pot_dev, pot, n*sizeof(double), cudaMemcpyHostToDevice));
  
  for (int i = 0; i < 20; i++) {
    std::cout << " " << i << std::endl;
    
    const int n_threads = _NTHREADS_;
    const int n_blocks = number_of_blocks(n_threads, n);
    test_pot<<<n_blocks, n_threads>>>(pot_dev, n);
    
    if(n_ranks > 0) {
      checkCudaErrors(cudaMemcpy(pot, pot_dev, n*sizeof(double), cudaMemcpyDeviceToHost));
      insist(MPI_Bcast(pot, n, MPI_DOUBLE, 0, MPI_COMM_WORLD) == MPI_SUCCESS);
      checkCudaErrors(cudaMemcpy(pot_dev, pot, n*sizeof(double), cudaMemcpyHostToDevice));
    }
  }
  
  if(pot_dev) { checkCudaErrors(cudaFree(pot_dev)); pot_dev = 0; }
}

void _cuda_mpi_test_for_quantum_dynamics_2(double *pot, const int n)
{
  int rank = -1;
  insist(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
  
  std::cout << " n = " << n << std::endl;
  
  double *pot_dev = 0;
  checkCudaErrors(cudaMallocManaged(&pot_dev, n*sizeof(double)));
  checkCudaErrors(cudaDeviceSynchronize());
  insist(pot_dev);
  
  memcpy(pot_dev, pot, n*sizeof(double));
  checkCudaErrors(cudaDeviceSynchronize());
  
  for (int i = 0; i < 20; i++) {
    std::cout << " " << i << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());
    
    const int n_threads =  _NTHREADS_;
    const int n_blocks = number_of_blocks(n_threads, n);
    test_pot<<<n_blocks, n_threads>>> (pot_dev, n);
    checkCudaErrors(cudaDeviceSynchronize());

    memcpy(pot, pot_dev, n*sizeof(double));
  }
  
  if(pot_dev) { checkCudaErrors(cudaFree(pot_dev)); pot_dev = 0; }
}


void cuda_mpi_test_for_quantum_dynamics(double *pot, const int n)
{
  std::cout << " CUDA MPI Test for Quantum Dynamics" << std::endl;

  _cuda_mpi_test_for_quantum_dynamics_1(pot, n);

}
