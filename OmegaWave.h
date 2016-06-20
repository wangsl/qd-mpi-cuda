
#ifndef OMEGA_WAVE_H
#define OMEGA_WAVE_H

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include <cufft.h>

#include "matlabArray.h"
#include "matlabStructures.h"
#include "complex.h"
#include "matlabUtils.h"
#include "cudaUtils.h"
#include "rmat.h"

class OmegaWaveFunction
{
public:
  
  OmegaWaveFunction() :
    omega(0), lmax(0),
    psi(0), r1(0), r2(0), theta(0),
    cufft_plan_for_psi(0), cublas_handle(0),
    psi_dev(0), pot_dev(0),
    associated_legendres_dev(0), 
    weighted_associated_legendres_dev(0),
    work_dev(0)
  { }
  
  ~OmegaWaveFunction();
  
  void setup_data(const int omega, const int lmax, const RMat &ass_leg,
		  Complex *psi_, const double *pot_dev,
		  const RadialCoordinate *r1, 
		  const RadialCoordinate *r2, 
		  const AngleCoordinate *theta,
		  const cufftHandle *cufft_plan_for_psi, 
		  const cublasHandle_t *cublas_handle);
  
  void test();

  double module() const;
  double potential_energy();
  
  void evolution_with_potential(const double *pot_dev, const double dt);
  void evolution_with_kinetic(const double dt);
  void evolution_with_rotational(const double dt);

private:
  
  Complex *psi;
  
  int omega, lmax;
  RMat associated_legendres;

  const RadialCoordinate *r1;
  const RadialCoordinate *r2;
  const AngleCoordinate *theta;
  
  const cufftHandle *cufft_plan_for_psi;
  const cublasHandle_t *cublas_handle;
  
  Complex *psi_dev;
  const double *pot_dev;
  Complex *associated_legendres_dev;
  Complex *weighted_associated_legendres_dev;
  Complex *work_dev;

  void setup_device_data();
  
  void setup_associated_legendres();
  void setup_weighted_associated_legendres();

  void forward_fft_for_psi();
  void backward_fft_for_psi(const int do_scale = 0);
};

#endif /* OMEGA_WAVE_H */
