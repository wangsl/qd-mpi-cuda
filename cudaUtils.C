
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "cudaUtils.h"

void cudaUtils::gpu_memory_usage()
{
  size_t free_byte = 0;
  size_t total_byte = 0;
  checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
  
  std::cout << " GPU memory usage:" 
	    << " used = " << (total_byte-free_byte)/1024.0/1024.0 << "MB,"
	    << " free = " << free_byte/1024.0/1024.0 << "MB,"
	    << " total = " << total_byte/1024.0/1024.0 << "MB" << std::endl;
}
