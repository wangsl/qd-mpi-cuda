
#include <iostream>
#include <mex.h>
#include <mpi.h>

void test_constant_memory();
void setup_constant_memory();

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  setup_constant_memory();

  test_constant_memory();
  

  //cuda_mpi_test();
  //void const_memory_test();
  //const_memory_test();
}
