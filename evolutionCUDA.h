
#ifndef EVOLUTION_CUDA_H
#define EVOLUTION_CUDA_H

#include <iostream>
#include "OmegaWave.h"
#include "matlabStructures.h"

class EvolutionCUDA
{
public:
  
  EvolutionCUDA(const double *pot,
		const RadialCoordinate &r1,
		const RadialCoordinate &r2,
		const AngleCoordinate &theta,
		OmegaStates &omegas,
		EvolutionTime &time
		);
  
  ~EvolutionCUDA();
  
  void test();

  Vec<OmegaWaveFunction> omega_psis;
  
private:
  const double *pot;
  
  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;
  OmegaStates &omegas;
  EvolutionTime &time;

  double *pot_dev;

  int has_setup_constant_memory;
  
  cufftHandle _cufft_plan_for_psi;
  int has_cufft_plan_for_psi;
  cufftHandle &cufft_plan_for_psi();

  cublasHandle_t _cublas_handle;
  int has_cublas_handle;
  cublasHandle_t &cublas_handle();
  
  // FFT for psi
  void setup_cufft_plan_for_psi();
  void destroy_cufft_plan_for_psi();

  void forward_fft_for_psi();
  void backward_fft_for_psi(const int do_scale = 0);
  
  // cublas handle
  void setup_cublas_handle();
  void destroy_cublas_handle();
  
  void setup_omega_psis();
  
  void allocate_device_data();
  void deallocate_device_data();

  void test_device();

  void evolution_with_potential(const double dt);
};

#endif/* EVOLUTION_CUDA_H */
