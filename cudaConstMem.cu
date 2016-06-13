
#include <iostream>
#include <cassert>
#include <helper_cuda.h>

// Total amount of constant memory:               65536 bytes

#define _MAX_MEM_SIZE_ 65300 

__constant__ char cuda_const_memory [_MAX_MEM_SIZE_];

#define _NTHREADS_ 512

inline int number_of_blocks(const int n_threads, const int n)
{ return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

void initialize(const char *x, const size_t size)
{
  checkCudaErrors(cudaMemcpyToSymbol(cuda_const_memory, x, size));
}

__device__ const double *dump1_dev()
{
  const double *ptr = (const double *) cuda_const_memory;
  return (const double *) &ptr[0];
}
 
__device__ const double *dump2_dev()
{
  const double *ptr = (const double *) cuda_const_memory;
  return (const double *) &ptr[10];
}

__global__ void show_x(const double *x, const int n)
{
  for(int i = 0; i < n; i++)
    printf("x[%d] = %.4f\n", i, x[i]);
}

__global__ void copy_to_y(double *y, const double *x, const int n)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    y[index] = x[index];
}

__global__ void copy_to_y_1(double *y, const int n)
{
  const double *x = dump1_dev();
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    y[index] = x[index];
}

__global__ void copy_to_y_2(double *y, const int n)
{
  const double *x = dump2_dev();
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    y[index] = x[index];
}

void const_memory_test()
{
  std::cout << " CUDA constant memory test" << std::endl;

  const int n1 = 10;
  const int n2 = 20;

  double *x = new double [n1+n2];
  assert(x);
  
  for (int i = 0; i < n1+n2; i++) 
    x[i] = i*0.1;
    
  initialize((char *) x, (n1+n2)*sizeof(double));

  memset(x, 0, (n1+n2)*sizeof(double));

  double *y = 0;
  checkCudaErrors(cudaMallocManaged(&y, (n1+n2)*sizeof(double)));

  const int n_threads = _NTHREADS_;
  int n_blocks = 0;

  n_blocks = number_of_blocks(n_threads, n1);
  copy_to_y_1<<<n_blocks, n_threads>>>(y, n1);
  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < n1; i++) 
    std::cout << i << " " << y[i] << std::endl;

  show_x<<<1, 1>>>(y, n1);

  n_blocks = number_of_blocks(n_threads, n2);
  copy_to_y_2<<<n_blocks, n_threads>>>(y, n2);
  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < n2; i++) 
    std::cout << i << " " << y[i] << std::endl;

  show_x<<<1, 1>>>(y, n2);

  if(y) { checkCudaErrors(cudaFree(y)); y = 0; }
  if(x) { delete [] x; x = 0; }
}
