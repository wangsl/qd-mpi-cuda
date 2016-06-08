
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


#include <mpi.h>
#include <iostream>
#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  int size = -1;
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  int rank = -1;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  
  std::cout << " " << size << "  " << rank << std::endl;
  
  const int bytes = size*sizeof(int);
  
  int *h_buff = new int [size];
  assert(h_buff);
  
  int *d_buff = 0;
  checkCudaErrors(cudaMalloc(&d_buff, bytes));
  
  int *d_rank = 0;
  checkCudaErrors(cudaMalloc(&d_rank, sizeof(int)));
  
  checkCudaErrors(cudaMemcpy(d_rank, &rank, sizeof(int), cudaMemcpyHostToDevice));
  
  // Preform Allgather using device buffer
  assert(MPI_Allgather(d_rank, 1, MPI_INT, d_buff, 1, MPI_INT, MPI_COMM_WORLD) == MPI_SUCCESS);
  
  // Check that the GPU buffer is correct
  checkCudaErrors(cudaMemcpy(h_buff, d_buff, bytes, cudaMemcpyDeviceToHost));
  
  std::cout << " ";
  for (int i = 0; i < size; i++) {
    std::cout << h_buff[i] << " "; 
    if (h_buff[i] != i) {
      std::cout << "Alltoall Failed!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  std::cout << std::endl;

  std::cout << " Success!" << std::endl;
  
  if(h_buff) { delete [] h_buff; h_buff = 0; }
  if(d_buff) { checkCudaErrors(cudaFree(d_buff)); d_buff = 0; }
  if(d_rank) { checkCudaErrors(cudaFree(d_rank)); d_rank = 0; }
}
