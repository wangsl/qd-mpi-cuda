
#include "OmegaWave.h"
#include "evolutionUtils.h"
#include "cudaUtils.h"

#include "evolutionCUDAaux.cu"

void OmegaWaveFunction::setup_data(const int omega_, const int lmax_, const RMat &leg_, 
				   Complex *psi_, 
				   const RadialCoordinate *r1_,
				   const RadialCoordinate *r2_,
				   const AngleCoordinate *theta_,
				   const cufftHandle *cufft_plan_for_psi_, 
				   const cublasHandle_t *cublas_handle_)
{
  omega = omega_;
  lmax = lmax_;
  ass_legendres = leg_;
  psi = psi_;

  r1 = r1_;
  r2 = r2_;
  theta = theta_;

  cufft_plan_for_psi = cufft_plan_for_psi_;
  cublas_handle = cublas_handle_;

  setup_device_data();
}

OmegaWaveFunction::~OmegaWaveFunction()
{
  std::cout << " Destruct omega wavepacket: " << omega << std::endl;

  psi = 0;
  r1 = 0;
  r2 = 0;
  theta = 0;
  
  cufft_plan_for_psi = 0;
  cublas_handle = 0;

  std::cout << " Deallocate device memory for wavepacket" << std::endl;
  _CUDA_FREE_(psi_dev);
  
  std::cout << " Deallocate device memory for associated Legendre Polynomials" << std::endl;
  _CUDA_FREE_(ass_legendres_dev);
  
  std::cout << " Deallocate device memory for work array" << std::endl;
  _CUDA_FREE_(work_dev);
}

void OmegaWaveFunction::setup_device_data()
{
  if(psi_dev && ass_legendres_dev) return;

  std::cout << " Omega: " << omega << std::endl;

  if(!psi_dev) {
    const int &n1 = r1->n;
    const int &n2 = r2->n;
    const int &n_theta = theta->n;
    std::cout << " Allocate device memory for wavepacket: " 
	      << n1 << " " << n2 << " " << n_theta << std::endl;
    const int size = n1*n2*n_theta;
    checkCudaErrors(cudaMalloc(&psi_dev, size*sizeof(Complex)));
    checkCudaErrors(cudaMemcpy(psi_dev, psi, size*sizeof(Complex), cudaMemcpyHostToDevice));
  }

  if(!ass_legendres_dev) {
    const int &n_theta = theta->n;
    const int nLeg = lmax - omega + 1;
    insist(nLeg > 0);
    std::cout << " Allocate device memory for associated Legendre Polynomials: " 
	      << n_theta << " " << nLeg << std::endl;
    const int size = n_theta*nLeg;
    checkCudaErrors(cudaMalloc(&ass_legendres_dev, size*sizeof(double)));
    checkCudaErrors(cudaMemcpy(ass_legendres_dev, (const double *) ass_legendres, 
			       size*sizeof(double), cudaMemcpyHostToDevice));
  }
  
  if(!work_dev) {
    const int &n1 = r1->n;
    const int &n2 = r2->n;
    const int &n_theta = theta->n;
    const int max_dim = n1*n2 + n_theta + 1024;
    std::cout << " Allocate device memory for work array: " << max_dim << std::endl;
    checkCudaErrors(cudaMalloc(&work_dev, max_dim*sizeof(Complex)));
  }
}

void OmegaWaveFunction::test()
{
  forward_fft_for_psi();
  backward_fft_for_psi(1);
}

void OmegaWaveFunction::forward_fft_for_psi()
{ 
  insist(cufft_plan_for_psi);
  
  insist(cufftExecZ2Z(*cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
                      CUFFT_FORWARD) == CUFFT_SUCCESS);
}

void OmegaWaveFunction::backward_fft_for_psi(const int do_scale)
{
  insist(cufft_plan_for_psi);
  
  insist(cufftExecZ2Z(*cufft_plan_for_psi, (cuDoubleComplex *) psi_dev, (cuDoubleComplex *) psi_dev, 
                      CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  if(do_scale) {
    insist(cublas_handle);
    const int &n1 = r1->n;
    const int &n2 = r2->n;
    const int &n_theta = theta->n;
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(*cublas_handle, n1*n2*n_theta, &s, (cuDoubleComplex *) psi_dev, 1) 
           == CUBLAS_STATUS_SUCCESS);
  }
}

void OmegaWaveFunction::evolution_with_potential(const double *pot_dev, const double dt)
{
  insist(pot_dev && psi_dev);
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int &n_theta = theta->n;
  const int n = n1*n2*n_theta;
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n);
  _evolution_with_potential_<<<n_blocks, n_threads>>>(psi_dev, pot_dev, n, dt);
}

double OmegaWaveFunction::module() const
{
  const double *w = theta->w;
  
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int &n_theta = theta->n;
  
  const cuDoubleComplex *psi_dev_ = (cuDoubleComplex *) psi_dev;
  double sum= 0.0;
  for(int k = 0; k < n_theta; k++) {
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(*cublas_handle, n1*n2, psi_dev_, 1, psi_dev_, 1, (cuDoubleComplex *) &dot)
           == CUBLAS_STATUS_SUCCESS);
    sum += w[k]*dot.real();
    psi_dev_ += n1*n2;
  }
  
  sum *= r1->dr*r2->dr;
  return sum;
}
