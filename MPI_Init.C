
#include <mex.h>
#include <mpi.h>

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  MPI_Init(0, 0);
  
  int n_procs = -100;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  
  int rank_id = -100;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(plhs[0]) = rank_id;
  
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(plhs[1]) = n_procs;
}
