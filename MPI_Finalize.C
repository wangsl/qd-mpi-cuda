
#include <mex.h>
#include <mpi.h>

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  MPI::Finalize();
}
