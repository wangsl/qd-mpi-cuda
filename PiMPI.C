
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <mex.h>
#include <mpi.h>

#include "matlabUtils.h"

inline double f(const double &x) 
{ 
  return 4.0/(1.0 + x*x);
}

inline double block_sum(const double &x1, const double &dx, const long &n_grids)
{
  double s = 0.0;
  for(long i = 0; i < n_grids; i++) {
    double x = x1 + i*dx;
    s += f(x);
  }
  
  return s*dx;
}

double calculate_pi()
{
  const double x1 = 0.0;
  const double x2 = 1.0;
  const long n = 2147483648;
  const double dx = (x2-x1)/(n-1);
  
  long n_grids = n;
  double xL = x1;
  
  int mpi_thread_id = -100;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_thread_id);
  
  int n_mpi_threads = -100;
  MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_threads);
  
  n_grids = n_grids/n_mpi_threads;
  xL = x1 + mpi_thread_id*n_grids*dx;
  
  if(mpi_thread_id == n_mpi_threads-1) {
    n_grids = n - mpi_thread_id*n_grids;
  }
  
  double s = block_sum(xL, dx, n_grids);
  
  double s_buf = 0.0;
  assert(MPI_Allreduce(&s, &s_buf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS);
  s = s_buf;
  
  double pi = s - 0.5*(f(0.0) + f(1.0))*dx;
  
  return pi;
}

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  for(int i = 0; i < 10; i++) {
    double pi =  calculate_pi();
    std::cout << std::setw(4) << i << " " << std::setw(16) << std::setprecision(14) << pi << std::endl;
    
    insist(!mexCallMATLAB(0, NULL, 0, NULL, "myTest"));
    
    insist(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  }
  
  std::cout.flush();  
}
