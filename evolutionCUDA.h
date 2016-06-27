
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

  int legendre_psi_max_size;

  double *pot_dev;

  int has_setup_constant_memory;

  void setup_legendre_psi_max_size();
  
  cufftHandle _cufft_plan_for_psi;
  int has_cufft_plan_for_psi;
  cufftHandle &cufft_plan_for_psi();

  cufftHandle _cufft_plan_for_legendre_psi;
  int has_cufft_plan_for_legendre_psi;
  cufftHandle &cufft_plan_for_legendre_psi();

  cublasHandle_t _cublas_handle;
  int has_cublas_handle;
  cublasHandle_t &cublas_handle();
  
  // FFT for psi
  void setup_cufft_plan_for_psi();
  void destroy_cufft_plan_for_psi();
  
  // FFT for legendre_psi
  void setup_cufft_plan_for_legendre_psi();
  void destroy_cufft_plan_for_legendre_psi();
  
  // cublas handle
  void setup_cublas_handle();
  void destroy_cublas_handle();
  
  void setup_omega_psis();
  
  void allocate_device_data();
  void deallocate_device_data();

  void test_device();

  void test_device_mpi();

  void evolution_with_potential(const double dt);
};

#endif/* EVOLUTION_CUDA_H */
