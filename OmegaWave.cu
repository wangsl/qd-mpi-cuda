
#include "OmegaWave.h"
#include "evolutionUtils.h"
#include "cudaUtils.h"

//extern __constant__ EvoltionUtils::RadialCoordinate r1_dev;
//extern __constant__ EvoltionUtils::RadialCoordinate r2_dev;

#include "evolutionCUDAaux.cu"

void OmegaWaveFunction::setup_data(const int n1_, const int n2_, const int n_theta_,
				   const int omega_, const int lmax_, const RMat &leg_, 
				   Complex *psi_)
{
  n1 = n1_;
  n2 = n2_;
  n_theta = n_theta_;
  omega = omega_;
  lmax = lmax_;
  ass_legendres = leg_;
  psi = psi_;

  setup_device_data();
}

OmegaWaveFunction::~OmegaWaveFunction()
{
  std::cout << " Destruct omega wavepacket: " << omega << std::endl;

  psi = 0;

  std::cout << " Deallocate device memory for wavepacket" << std::endl;
  _CUDA_FREE_(psi_dev);
  
  std::cout << " Deallocate device memory for associated Legendre Polynomials" << std::endl;
  
  _CUDA_FREE_(ass_legendres_dev);
}

void OmegaWaveFunction::setup_device_data()
{
  if(psi_dev && ass_legendres_dev) return;

  std::cout << " Omega: " << omega << std::endl;

  if(!psi_dev) {
    std::cout << " Allocate device memory for wavepacket: " << n1 << " " << n2 << " " << n_theta << std::endl;
    const int size = n1*n2*n_theta;
    checkCudaErrors(cudaMalloc(&psi_dev, size*sizeof(Complex)));
    checkCudaErrors(cudaMemcpy(psi_dev, psi, size*sizeof(Complex), cudaMemcpyHostToDevice));
  }

  if(!ass_legendres_dev) {
    const int nLeg = lmax - omega + 1;
    insist(nLeg > 0);
    std::cout << " Allocate device memory for associated Legendre Polynomials: " << n_theta << " " << nLeg << std::endl;
    const int size = n_theta*nLeg;
    checkCudaErrors(cudaMalloc(&ass_legendres_dev, size*sizeof(double)));
    checkCudaErrors(cudaMemcpy(ass_legendres_dev, (const double *) ass_legendres, 
			       size*sizeof(double), cudaMemcpyHostToDevice));
  }

#if 0
  _test_constant_memory_<<<1,1>>>();
  //_print_gauss_legendre_weight_<<<1,1>>>(n_theta);

  double *dump_dev = 0;
  int n_threads = 0;
  int n_blocks = 0;
  
  checkCudaErrors(cudaMalloc(&dump_dev, n1*sizeof(double)));
  n_threads = _NTHREADS_;
  n_blocks = cudaUtils::number_of_blocks(n_threads, n1);
  _calculate_dump_function_<<<n_blocks, n_threads>>>(dump_dev, 1);
  _show_dump_function_<<<1,1>>>(dump_dev, 1);
  _CUDA_FREE_(dump_dev);

  checkCudaErrors(cudaMalloc(&dump_dev, n2*sizeof(double)));
  n_threads = _NTHREADS_;
  n_blocks = cudaUtils::number_of_blocks(n_threads, n2);
  _calculate_dump_function_<<<n_blocks, n_threads>>>(dump_dev, 2);
  _show_dump_function_<<<1,1>>>(dump_dev, 2);
  _CUDA_FREE_(dump_dev);
#endif
}
