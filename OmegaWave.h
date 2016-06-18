
#ifndef OMEGA_WAVE_H
#define OMEGA_WAVE_H

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include <cufft.h>

#include "matlabArray.h"
#include "complex.h"
#include "matlabUtils.h"
#include "cudaUtils.h"
#include "rmat.h"

class OmegaWaveFunction
{
public:
  OmegaWaveFunction() :
    n1(0), n2(0), n_theta(0), omega(0), lmax(0),
    psi(0), 
    psi_dev(0), ass_legendres_dev(0),
    has_cublas_handle(0), has_cufft_plan_for_psi(0)
  { }
  
  ~OmegaWaveFunction();

  void setup_data(const int n1, const int n2, const int n_theta,
		  const int omega, const int lmax, const RMat &leg,
		  Complex *psi);
  
  void test();

  void test(cufftHandle &cufft_plan_for_psi, cublasHandle_t &cublas_handle);

private:
  Complex *psi;
  
  int n1, n2, n_theta;
  int omega, lmax;
  RMat ass_legendres;
  
  Complex *psi_dev;
  double *ass_legendres_dev;

  cublasHandle_t cublas_handle;
  int has_cublas_handle;

  cufftHandle cufft_plan_for_psi;
  int has_cufft_plan_for_psi;

  //cufftHandle cufft_plan_for_legendre_psi;

  void setup_device_data();

  // cublas handle
  void setup_cublas_handle();
  void destroy_cublas_handle();
  
  // FFT for psi
  void setup_cufft_plan_for_psi();
  void destroy_cufft_plan_for_psi();
  void forward_fft_for_psi();
  void backward_fft_for_psi(const int do_scale = 0);
  
  void forward_fft_for_psi(cufftHandle &cufft_plan_for_psi);
  void backward_fft_for_psi(cufftHandle &cufft_plan_for_psi, cublasHandle_t &cublas_handle, 
			    const int do_scale = 0);
};

#endif /* OMEGA_WAVE_H */
