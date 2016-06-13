
#include <iostream>
#include <mex.h>
#include <mpi.h>

//void cuda_mpi_test();

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  //cuda_mpi_test();
  void const_memory_test();
  const_memory_test();
}
