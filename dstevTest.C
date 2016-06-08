
#include <iostream>
#include <mex.h>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "rmat.h"
#include "fort.h"

/*
  Refercens:
  J. Phys. Chem. A, 102, 9372-9379 (1998)
  John Zhang  Theory and Application of Quantum Molecular Dynamics P343
  http://www.netlib.org/lapack/explore-html/d7/d48/dstev_8f.html
*/

extern "C" void FORT(dstev)(const char *JOBZ, const FInt &N, double *D, double *E, double *Z,
			    const FInt &LDZ, double *work, FInt &info);

inline double lambda(const int J, const int Omega, const int sign)
{
  double c = 0.0;
  if(J >= Omega) {
    c = sign%2 == 0 ? 
      sqrt(J*(J+1.0) - Omega*(Omega+1.0)) :
      sqrt(J*(J+1.0) - Omega*(Omega-1.0)) ;
  }
  
  return c;
}

inline int kronecker_delta(const int &a, const int &b)
{
  return a == b ? 1 : 0;
}

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray **prhs)
{
  std::cout.precision(15);

  std::cout << " DSTEV Test" << std::endl;

  const int *in_data = (int *) mxGetPr(prhs[0]);
  assert(in_data);

  const int &J = in_data[0];
  const int &M = in_data[1];
  const int &p = in_data[2];
  const int &j = in_data[3];
  
  std::cout << " " << J << " " << M << " " << p << " " << j << std::endl;

  const int Omega_min = (J+p)%2 == 0 ? 0: 1;
  const int Omega_max = J;
  const int n = Omega_max - Omega_min + 1;

  std::cout << " n = " << n << std::endl
	    << " Omega_min = " << Omega_min << std::endl;
  
  double *diag_eles = new double [n];
  assert(diag_eles);
  
  for(int k = 0; k < n; k++) {
    const int K = k + Omega_min;
    diag_eles[k] = J*(J+1.0) + j*(j+1.0) - 2.0*K*K;
  }
  
#if 0
  for(int i = 0; i < n; i++)
    std::cout << i << " " << diag_eles[i] << std::endl;
#endif
  
  double *sub_diag_eles = new double [n-1];
  assert(sub_diag_eles);
  
  for(int k = 0; k < n-1; k++) {
    const int K = k + Omega_min;
    sub_diag_eles[k] = -lambda(J, K, 0)*lambda(j, K, 0)*sqrt(1.0 + kronecker_delta(K, 0));
  }
  
#if 0
  for(int i = 0; i < n-1; i++)
    std::cout << i << " " << sub_diag_eles[i] << std::endl;
#endif

  RMat evs(n, n);
  double *work = new double [std::max(1, 2*n-2)];
  assert(work);

  FInt n_ = n;
  const char jobV[128] = "V";
  FInt info = -100;
  FORT(dstev)(jobV, n_, diag_eles, sub_diag_eles, evs, n_, work, info);
  assert(info == 0);  

#if 0
  for(int i = 0; i < n; i++)
    std::cout << " " << i << " " << diag_eles[i] << std::endl;

  std::cout << evs << std::endl;

  std::cout << std::endl;
  
  std::cout << std::endl;
  for(int i = 0; i < n; i++)
    std::cout << i << " " << evs(i, n-1) << std::endl;
#endif

  if(in_data) { in_data = 0; }
  if(diag_eles) { delete [] diag_eles; diag_eles = 0; }
  if(sub_diag_eles) { delete [] sub_diag_eles; sub_diag_eles = 0; }
  if(work) { delete [] work; work = 0; }
}

