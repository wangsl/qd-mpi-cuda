

#ifndef CUDA_CONST_MEM__H
#define CUDA_CONST_MEM__H

#define _MAX_MEM_SIZE_ 65000 

class CUDAConstMemory 
{
public:
  
  CUDAConstMemory() { }

  ~CUDAConstMemory() 
  { 
    ptr = 0;
    dump1_dev = 0;
    dump2_dev = 0;
    mem_size = 0;
  }

  size_t mem_size;
  double *dump1_dev;
  double *dump2_dev;
  
  void initialize(const char *x, const size_t size);

private:
  char *ptr;
  static const int max_size = _MAX_MEM_SIZE_;

};

#endif /* CUDA_CONST_MEM_H */


