
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define _CUDA_FREE_(x) if(x) { checkCudaErrors(cudaFree(x)); x = 0; }

#define _NTHREADS_ 512

namespace cudaUtils {
  
  inline int number_of_blocks(const int n_threads, const int n)
  { return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

  void gpu_memory_usage();
}

#endif /* MY_CUDA_UTILS_H */


