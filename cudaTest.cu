
#include <iostream>
#include <mex.h>
#include <mpi.h>
#include <helper_cuda.h>

__global__ void AplusB(int *ret, const int a, const int b)
{
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void _cuda_mpi_test_1()
{
  std::cout << " MPI Test 1" << std::endl;

  int rank = -1;
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

  int *ret = 0;
  checkCudaErrors(cudaMallocManaged(&ret, 1000*sizeof(int)));
  assert(ret);
  checkCudaErrors(cudaMemset(ret, 0, 1000*sizeof(int)));
  checkCudaErrors(cudaDeviceSynchronize());
  
  if(rank == 0) {
    AplusB<<<1, 1000>>>(ret, 10, 1000);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  assert(MPI_Bcast(ret, 1000, MPI_INT, 0, MPI_COMM_WORLD) == MPI_SUCCESS);

  if(rank != 0) {
    AplusB<<<1, 1000>>>(ret, 10, 1000);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  for(int i = 0; i < 10; i++)
    std::cout << " " << i << "  " << ret[i] << std::endl;

  if(ret) { checkCudaErrors(cudaFree(ret)); ret = 0; }
  
  return;
}

void _cuda_mpi_test_2()
{
  std::cout << " MPI Test 2" << std::endl;

  int rank = -1;
  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

  int *ret_d = 0;
  checkCudaErrors(cudaMalloc(&ret_d, 1000*sizeof(int)));
  assert(ret_d);
  checkCudaErrors(cudaMemset(ret_d, 0, 1000*sizeof(int)));
  
  if(rank == 0) 
    AplusB<<<1, 1000>>>(ret_d, 10, 1000);
  
  assert(MPI_Bcast(ret_d, 1000, MPI_INT, 0, MPI_COMM_WORLD) == MPI_SUCCESS);
  
  if(rank != 0)
    AplusB<<<1, 1000>>>(ret_d, 10, 1000);
  
  int *ret_h = new int [1000];
  assert(ret_h);
  checkCudaErrors(cudaMemcpy(ret_h, ret_d, 1000*sizeof(int), cudaMemcpyDeviceToHost));
  
  if(ret_d) { checkCudaErrors(cudaFree(ret_d)); ret_d = 0; }
  
  for(int i = 0; i < 10; i++)
    std::cout << " " << i << "  " << ret_h[i] << std::endl;
  
  if(ret_h) { delete [] ret_h; ret_h = 0; }
  
  return;
}

void cuda_mpi_test()
{
  std::cout << " CUDA MPI Test" << std::endl;

  _cuda_mpi_test_1();

  _cuda_mpi_test_2();
}

