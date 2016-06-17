
#include <helper_cuda.h>
#include "evolutionCUDA.h"
#include "evolutionUtils.h"

// defined as extern in evolutionUtils.h

__constant__ EvoltionUtils::RadialCoordinate r1_dev;
__constant__ EvoltionUtils::RadialCoordinate r2_dev;
__constant__ double gauss_legendre_weight_dev[512];

#include "evolutionCUDAaux.cu"

void EvolutionCUDA::allocate_device_data()
{ 
  if(!pot_dev) {
    std::cout << " Allocate device memory for potential" << std::endl;
    const int &n1 = r1.n;
    const int &n2 = r2.n; 
    const int &n_theta = theta.n;
    const size_t size = n1*n2*n_theta;
    checkCudaErrors(cudaMalloc(&pot_dev, size*sizeof(double)));
    insist(pot_dev);
    checkCudaErrors(cudaMemcpy(pot_dev, pot, size*sizeof(double), cudaMemcpyHostToDevice));
  }
  
  if(!has_setup_constant_memory) {
    std::cout << " Setup device constant memory" << std::endl;
    const int &n_theta = theta.n;
    size_t size = 0;
    checkCudaErrors(cudaGetSymbolSize(&size, gauss_legendre_weight_dev));
    insist(size/sizeof(double) > n_theta);
    checkCudaErrors(cudaMemcpyToSymbol(gauss_legendre_weight_dev, theta.w, n_theta*sizeof(double)));
    
    EvoltionUtils::copy_radial_coordinate_to_device(r1_dev, r1.left, r1.dr, r1.mass,
						    r1.dump_Cd, r1.dump_xd, r1.n);
    EvoltionUtils::copy_radial_coordinate_to_device(r2_dev, r2.left, r2.dr, r2.mass, 
						    r2.dump_Cd, r2.dump_xd, r2.n);
    has_setup_constant_memory = 1;
  }
}

void EvolutionCUDA::deallocate_device_data()
{
  std::cout << " Deallocate device memory: potential" << std::endl;
  _CUDA_FREE_(pot_dev);
}

void EvolutionCUDA::setup_Omega_Psis()
{
  const Vec<int> omgs = omegas.omegas;
  const Vec<RMat> &ass_legendres = omegas.associated_legendres;	
  Vec<Vec<Complex> > &wave_packets = omegas.wave_packets;
  
  const int n_omgs = omgs.size();
  
  Omega_Psis.resize(n_omgs);
  
  for (int i = 0; i < n_omgs; i++)
    Omega_Psis[i].setup_data(r1.n, r2.n, theta.n, omgs[i], omegas.lmax,
			     ass_legendres[i], (Complex *) wave_packets[i]);

}

void EvolutionCUDA::test_device()
{
  //std::cout << " Test CUDA data" << std::endl;
  
  //_test_constant_memory_<<<1,1>>>();
  
  //_print_gauss_legendre_weight_<<<1,1>>>(theta.n);
  
  double *dump1 = 0;
  checkCudaErrors(cudaMalloc(&dump1, r1.n*sizeof(double)));
  
  int n_threads = _NTHREADS_;
  int n_blocks = cudaUtils::number_of_blocks(n_threads, r1.n);
  _calculate_dump_function_<<<n_blocks, n_threads>>>(dump1, 1);

  _CUDA_FREE_(dump1);
  
  double *dump2 = 0;
  checkCudaErrors(cudaMalloc(&dump2, r2.n*sizeof(double)));
  
  n_threads = _NTHREADS_;
  n_blocks = cudaUtils::number_of_blocks(n_threads, r2.n);
  _calculate_dump_function_<<<n_blocks, n_threads>>>(dump2, 2);

  _CUDA_FREE_(dump2);
}
