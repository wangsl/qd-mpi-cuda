
#include "OmegaWave.h"
#include "evolutionUtils.h"
#include "cudaUtils.h"

#include "evolutionCUDAaux.cu"

void OmegaWaveFunction::setup_data(const int omega_, const int lmax_, const RMat &ass_leg_, 
				   Complex *psi_, const double *pot_dev_,
				   const RadialCoordinate *r1_,
				   const RadialCoordinate *r2_,
				   const AngleCoordinate *theta_,
				   const cufftHandle *cufft_plan_for_psi_, 
				   const cublasHandle_t *cublas_handle_)
{
  omega = omega_;
  lmax = lmax_;
  associated_legendres = ass_leg_;

  psi = psi_;
  
  pot_dev = pot_dev_;

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

  pot_dev = 0;
  
  cufft_plan_for_psi = 0;
  cublas_handle = 0;

  std::cout << " Deallocate device memory for wavepacket" << std::endl;
  _CUDA_FREE_(psi_dev);
  
  std::cout << " Deallocate device memory for complex associated Legendre Polynomials" << std::endl;
  _CUDA_FREE_(associated_legendres_dev);

  std::cout << " Deallocate device memory for weighted complex associated Legendre Polynomials" << std::endl;
  _CUDA_FREE_(weighted_associated_legendres_dev);
  
  std::cout << " Deallocate device memory for work array" << std::endl;
  _CUDA_FREE_(work_dev);
}

void OmegaWaveFunction::setup_device_data()
{
  if(psi_dev && associated_legendres_dev) return;

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

  setup_associated_legendres();
  setup_weighted_associated_legendres();
  
  if(!work_dev) {
    const int &n1 = r1->n;
    const int &n2 = r2->n;
    const int &n_theta = theta->n;
    const int max_dim = n1*n2 + n_theta + 1024;
    std::cout << " Allocate device memory for work array: " << max_dim << std::endl;
    checkCudaErrors(cudaMalloc(&work_dev, max_dim*sizeof(Complex)));
  }
}

void OmegaWaveFunction::setup_associated_legendres()
{
  if(associated_legendres_dev) return;
  
  const int &n_theta = theta->n;
  const int n_legs = lmax - omega + 1;
  insist(n_legs > 0);
  
  const RMat &p = associated_legendres;
  insist(p.rows() == n_theta);
  
  Mat<Complex> p_complex(n_legs, n_theta);
  for(int l = 0; l < n_legs; l++) {
    for(int k = 0; k < n_theta; k++) {
      p_complex(l,k) = Complex(p(k,l), 0.0);
    }
  }
  
  std::cout << " Allocate device memory for complex associated Legendre Polynomials: " 
	    << n_legs << " " << n_theta << std::endl;
  
  const int size = n_legs*n_theta;
  checkCudaErrors(cudaMalloc(&associated_legendres_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemcpy(associated_legendres_dev, (const Complex *) p_complex,
			     size*sizeof(Complex), cudaMemcpyHostToDevice));
}

void OmegaWaveFunction::setup_weighted_associated_legendres()
{
  if(weighted_associated_legendres_dev) return;
  
  const int &n_theta = theta->n;
  const int n_legs = lmax - omega + 1;
  insist(n_legs > 0);
  
  const double *w = theta->w;
  
  const RMat &p = associated_legendres;
  insist(p.rows() == n_theta);
  
  Mat<Complex> wp_complex(n_theta, n_legs);
  for(int l = 0; l < n_legs; l++) {
    for(int k = 0; k < n_theta; k++) {
      wp_complex(k,l) = Complex(w[k]*p(k,l), 0.0);
    }
  }
  
  std::cout << " Allocate device memory for weighted complex associated Legendre Polynomials: " 
	    << n_theta << " " << n_legs << std::endl;
  
  const int size = n_theta*n_legs;
  checkCudaErrors(cudaMalloc(&weighted_associated_legendres_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemcpy(weighted_associated_legendres_dev, (const Complex *) wp_complex,
			     size*sizeof(Complex), cudaMemcpyHostToDevice));
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

double OmegaWaveFunction::potential_energy()
{
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int &n_theta = theta->n;
  
  const double *w = theta->w;
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);
  
  const cuDoubleComplex *psi_dev_ = (cuDoubleComplex *) psi_dev;
  const double *pot_dev_ = pot_dev;
  double sum = 0.0;
  for(int k = 0; k < n_theta; k++) {
    cudaMath::_vector_multiplication_<Complex, Complex, double><<<n_blocks, n_threads>>>
      ((Complex *) psi_tmp_dev, (const Complex *) psi_dev_, pot_dev_, n1*n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(*cublas_handle, n1*n2, psi_dev_, 1, psi_tmp_dev, 1,
                       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    sum += w[k]*dot.real();
    
    psi_dev_ += n1*n2;
    pot_dev_ += n1*n2;
  }

  sum *= r1->dr*r2->dr;
  return sum;
}
