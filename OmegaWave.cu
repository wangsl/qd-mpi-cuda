
#include "OmegaWave.h"

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
}
