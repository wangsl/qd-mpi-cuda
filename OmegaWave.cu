
#include "OmegaWave.h"
#include "evolutionUtils.h"
#include "cudaUtils.h"

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
  setup_cublas_handle();
}

OmegaWaveFunction::~OmegaWaveFunction()
{
  std::cout << " Destruct omega wavepacket: " << omega << std::endl;

  psi = 0;

  std::cout << " Deallocate device memory for wavepacket" << std::endl;
  _CUDA_FREE_(psi_dev);
  
  std::cout << " Deallocate device memory for associated Legendre Polynomials" << std::endl;
  
  _CUDA_FREE_(ass_legendres_dev);

  destroy_cublas_handle();
  destroy_cufft_plan_for_psi();
}

void OmegaWaveFunction::setup_device_data()
{
  if(psi_dev && ass_legendres_dev) return;

  std::cout << " Omega: " << omega << std::endl;

  if(!psi_dev) {
    std::cout << " Allocate device memory for wavepacket: " 
	      << n1 << " " << n2 << " " << n_theta << std::endl;
    const int size = n1*n2*n_theta;
    checkCudaErrors(cudaMalloc(&psi_dev, size*sizeof(Complex)));
    checkCudaErrors(cudaMemcpy(psi_dev, psi, size*sizeof(Complex), cudaMemcpyHostToDevice));
  }

  if(!ass_legendres_dev) {
    const int nLeg = lmax - omega + 1;
    insist(nLeg > 0);
    std::cout << " Allocate device memory for associated Legendre Polynomials: " 
	      << n_theta << " " << nLeg << std::endl;
    const int size = n_theta*nLeg;
    checkCudaErrors(cudaMalloc(&ass_legendres_dev, size*sizeof(double)));
    checkCudaErrors(cudaMemcpy(ass_legendres_dev, (const double *) ass_legendres, 
			       size*sizeof(double), cudaMemcpyHostToDevice));
  }
}

void OmegaWaveFunction::setup_cublas_handle()
{
  if(has_cublas_handle) return;
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 1;
}

void OmegaWaveFunction::destroy_cublas_handle()
{
  if(!has_cublas_handle) return;
  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 0;
}

void OmegaWaveFunction::setup_cufft_plan_for_psi()
{
  if(has_cufft_plan_for_psi) return;
  
  const int dim [] = { n2, n1 };
  
  insist(cufftPlanMany(&cufft_plan_for_psi, 2, const_cast<int *>(dim), NULL, 1, n1*n2, NULL, 1, n1*n2,
                       CUFFT_Z2Z, n_theta) == CUFFT_SUCCESS);
  
  has_cufft_plan_for_psi = 1;
}

void OmegaWaveFunction::destroy_cufft_plan_for_psi()
{
  if(!has_cufft_plan_for_psi) return;
  insist(cufftDestroy(cufft_plan_for_psi) == CUFFT_SUCCESS);
  has_cufft_plan_for_psi = 0;
}

void OmegaWaveFunction::forward_fft_for_psi()
{ 
  setup_cufft_plan_for_psi();
  
  insist(cufftExecZ2Z(cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
                      CUFFT_FORWARD) == CUFFT_SUCCESS);
}

void OmegaWaveFunction::backward_fft_for_psi(const int do_scale)
{
  setup_cufft_plan_for_psi();
  
  insist(cufftExecZ2Z(cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
                      CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  if(do_scale) {
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
           == CUBLAS_STATUS_SUCCESS);
  }
}

void OmegaWaveFunction::test()
{
  forward_fft_for_psi();
  backward_fft_for_psi(1);
}
