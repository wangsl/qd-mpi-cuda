
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
		OmegaStates &omegas
		);
  
  ~EvolutionCUDA();
  
  void test();

  Vec<OmegaWaveFunction> Omega_Psis;
  
private:
  const double *pot;
  
  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;
  OmegaStates &omegas;

  int has_setup_constant_memory;

  double *pot_dev;

  void setup_Omega_Psis();

  void allocate_device_data();
  void deallocate_device_data();

  void test_device();
};

#endif/* EVOLUTION_CUDA_H */
