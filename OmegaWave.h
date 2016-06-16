
#ifndef OMEGA_WAVE_H
#define OMEGA_WAVE_H

#include <helper_cuda.h>
#include "matlabArray.h"
#include "complex.h"
#include "matlabUtils.h"
#include "cudaUtils.h"
#include "rmat.h"

class OmegaWaveFunction
{
public:
  OmegaWaveFunction() :
    psi(0), 
    psi_dev(0), ass_legendres_dev(0)
  { }
  
  ~OmegaWaveFunction();

  void setup_data(const int n1, const int n2, const int n_theta,
		  const int omega, const int lmax, const RMat &leg,
		  Complex *psi);

private:
  Complex *psi;
  
  int n1, n2, n_theta;
  int omega, lmax;
  RMat ass_legendres;
  
  Complex *psi_dev;
  double *ass_legendres_dev;

  void setup_device_data();
};

#endif /* OMEGA_WAVE_H */
