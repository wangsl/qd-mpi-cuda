
#ifndef EVOLUTION_MPI_CUDA_H
#define EVOLUTION_MPI_CUDA_H

#include <iostream>
#include "OmegaWave.h"
#include "matlabStructures.h"

class EvolutionMPICUDA
{
public:
  
  EvolutionMPICUDA(const MatlabArray<double> &m_pot,
		   MatlabArray<Complex> &m_psi,
		   const RadialCoordinate &r1,
		   const RadialCoordinate &r2,
		   const AngleCoordinate &theta
		   );

  ~EvolutionMPICUDA();

  void test();

private:
  double *pot;

  const MatlabArray<double> &m_pot;
  MatlabArray<Complex> &m_psi;

  const RadialCoordinate &r1;
  const RadialCoordinate &r2;
  const AngleCoordinate &theta;

  double *pot_dev;

  void allocate_device_data();
  void deallocate_device_data();

  void test_device();
};

#endif/* EVOLUTION_MPI_CUDA_H */
