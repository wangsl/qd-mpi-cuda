
#include <helper_cuda.h>
#include "evolutionMPICUDA.h"
#include "evolutionUtils.h"

__constant__ EvoltionUtils::RadialCoordinate r1_dev;
__constant__ EvoltionUtils::RadialCoordinate r2_dev;

__constant__ double gauss_legendre_weight_dev[512];

#include "evolutionUtils.cu"

void EvolutionMPICUDA::allocate_device_data()
{ 
  std::cout << " Allocate device memory" << std::endl;
  
  const int &n_dims = m_pot.n_dims();
  insist(n_dims == 3);
  
  const size_t *dims = m_pot.dims();
  
  const int &n1 = dims[0];
  const int &n2 = dims[1];
  const int &n_theta = dims[2];

  const int n = n1*n2*n_theta;
  
  std::cout << " Wavepacket size: " << n1 << " " << n2 << " " << n_theta << std::endl;
  
  if(!pot_dev) {
    checkCudaErrors(cudaMalloc(&pot_dev, n*sizeof(double)));
    insist(pot_dev);
    checkCudaErrors(cudaMemcpy(pot_dev, pot, n*sizeof(double), cudaMemcpyHostToDevice));
  }
  
  if(!psi_dev) {
    checkCudaErrors(cudaMalloc(&psi_dev, n*sizeof(Complex)));
    checkCudaErrors(cudaMemcpy(psi_dev, psi, n*sizeof(Complex), cudaMemcpyHostToDevice));
  }

  EvoltionUtils::copy_radial_coordinate_to_device(r1_dev, r1.left, r1.dr, r1.mass, r1.n);
  EvoltionUtils::copy_radial_coordinate_to_device(r2_dev, r2.left, r2.dr, r2.mass, r2.n);

}

void EvolutionMPICUDA::deallocate_device_data()
{
  std::cout << " DeAllocate device memory" << std::endl;

#define _CUDA_FREE_(x) if(x) { checkCudaErrors(cudaFree(x)); x = 0; }

  _CUDA_FREE_(pot_dev);
  _CUDA_FREE_(psi_dev);

#undef _CUDA_FREE_
}

void EvolutionMPICUDA::test_device()
{
  std::cout << " Test CUDA data" << std::endl;

  _test_dev_r<<<1,1>>>();
}
