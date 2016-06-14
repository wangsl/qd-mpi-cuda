
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define _NTHREADS_ 512

namespace cudaUtils {
  
  inline int number_of_blocks(const int n_threads, const int n)
  { return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }
}


#endif /* CUDA_UTILS_H */


