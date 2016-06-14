
#ifndef OMEGA_WAVE_H
#define OMEGA_WAVE_H

#include <helper_cuda.h>
#include "matlabArray.h"
#include "complex.h"
#include "matlabUtils.h"
#include "cudaUtils.h"

class OmegaWaveFunction
{
public:
  OmegaWaveFunction() :
    m_psi(0), psi(0), psi_dev(0)
  { }
  
  ~OmegaWaveFunction();

  void setup_data(MatlabArray<Complex> &m_psi);

private:
  MatlabArray<Complex> *m_psi;
  Complex *psi;

  Complex *psi_dev;

  void setup_device_data();
};

#endif /* OMEGA_WAVE_H */
