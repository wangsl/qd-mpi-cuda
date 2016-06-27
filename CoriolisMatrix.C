
#include <iostream>
//#include <mex.h>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "rmat.h"
#include "fort.h"

#include "evolutionUtils.h"

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

void setup_coriolis_matrix(const int J, const int p, const int j)
{
  if(J == 0) insist(p == 0);
  
  std::cout << " J: " << J << " p: " << p << " j: " << j << std::endl;
  
  const int Omega_min = (J+p)%2 == 0 ? 0: 1;
  const int Omega_max = std::min(J, j);
  const int n = Omega_max - Omega_min + 1;

  std::cout << " Omega_min: " << Omega_min << " Omega_max: " << Omega_max 
	    << " size: " << n << std::endl;
  
  double *diag_eles = new double [n];
  insist(diag_eles);
  
  for(int k = 0; k < n; k++) {
    const int Omega = k + Omega_min;
    diag_eles[k] = J*(J+1.0) + j*(j+1.0) - 2.0*Omega*Omega;
  }
  
#if 0
  std::cout << " Diagonals: ";
  for(int i = 0; i < n; i++)
    std::cout << diag_eles[i] << " ";
  std::cout << endl;
#endif
  
  double *sub_diag_eles = new double [n-1];
  insist(sub_diag_eles);
  for(int k = 0; k < n-1; k++) {
    const int Omega = k + Omega_min;
    sub_diag_eles[k] = -lambda(J, Omega, 0)*lambda(j, Omega, 0)*sqrt(1.0 + kronecker_delta(Omega, 0));
  }
  
#if 0
  std::cout << " SubDiagonals: ";
  for(int i = 0; i < n-1; i++)
    std::cout << sub_diag_eles[i] << " ";
  std::cout << std::endl;
#endif

  RMat evs(n, n);
  double *work = new double [std::max(1, 2*n-2)];
  insist(work);

  FInt n_ = n;
  const char jobV[32] = "V";
  FInt info = -100;
  FORT(dstev)(jobV, n_, diag_eles, sub_diag_eles, evs, n_, work, info);
  assert(info == 0);  

#if 0
  RVec d_(n, diag_eles);
  d_.show_in_one_line();

  std::cout << evs << std::endl;
#endif

  if(diag_eles) { delete [] diag_eles; diag_eles = 0; }
  if(sub_diag_eles) { delete [] sub_diag_eles; sub_diag_eles = 0; }
  if(work) { delete [] work; work = 0; }
}

