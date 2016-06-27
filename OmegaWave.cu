
#include <mpi.h>

#include "OmegaWave.h"
#include "evolutionUtils.h"
#include "cudaUtils.h"

#include "evolutionCUDAaux.cu"

void OmegaWaveFunction::setup_data(const int omega_, const int lmax_, 
				   const int legendre_psi_max_size_,
				   const RMat &ass_leg_, 
				   Complex *psi_, const double *pot_dev_,
				   const RadialCoordinate *r1_,
				   const RadialCoordinate *r2_,
				   const AngleCoordinate *theta_,
				   const cufftHandle *cufft_plan_for_psi_, 
				   const cufftHandle *cufft_plan_for_legendre_psi_, 
				   const cublasHandle_t *cublas_handle_)
{
  omega = omega_;
  lmax = lmax_;
  legendre_psi_max_size = legendre_psi_max_size_;
  
  associated_legendres = ass_leg_;

  psi = psi_;
  
  pot_dev = pot_dev_;

  r1 = r1_;
  r2 = r2_;
  theta = theta_;

  cufft_plan_for_psi = cufft_plan_for_psi_;
  cufft_plan_for_legendre_psi = cufft_plan_for_legendre_psi_;
  
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
  cufft_plan_for_legendre_psi = 0;
  cublas_handle = 0;

  if(psi_dev) {
    std::cout << " Deallocate device memory for psi" << std::endl;
    _CUDA_FREE_(psi_dev);
  }

  if(legendre_psi_dev) {
    std::cout << " Deallocate device memory for Legendre psi" << std::endl;
    _CUDA_FREE_(legendre_psi_dev);
  }
  
  if(associated_legendres_dev) {
    std::cout << " Deallocate device memory for complex associated Legendre Polynomials" << std::endl;
    _CUDA_FREE_(associated_legendres_dev);
  }

  if(weighted_associated_legendres_dev) {
    std::cout << " Deallocate device memory for weighted complex associated Legendre Polynomials" << std::endl;
    _CUDA_FREE_(weighted_associated_legendres_dev);
  }

  if(work_dev) {
    std::cout << " Deallocate device memory for work array" << std::endl;
    _CUDA_FREE_(work_dev);
  }
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
  forward_legendre_transform();

  const double mod = module_for_legendre_psi();

  std::cout << " module_for_legendre_psi: " << mod << std::endl;

  forward_fft_for_legendre_psi();

  const double e_kin = kinetic_energy_for_legendre_psi(0);

  std::cout << " kinetic_energy_for_legendre_psi: " << e_kin << " " << e_kin/mod << std::endl;

  backward_fft_for_legendre_psi(1);

  std::cout << " module_for_legendre_psi: " << module_for_legendre_psi() << std::endl;

  backward_legendre_transform();

  std::cout << " module_for_psi: " << module_for_psi() << std::endl;

  const double e_pot = potential_energy();
  std::cout << " potential_energy: " << e_pot << " " << e_pot/mod << std::endl;
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

double OmegaWaveFunction::module_for_psi() const
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

void OmegaWaveFunction::setup_legendre_psi()
{
  if(legendre_psi_dev) return;

  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int n_legs = lmax - omega + 1;

  insist(n_legs <= legendre_psi_max_size);

  std::cout << " Allocate device memory for Legendre psi: "
	    << n1 << " " << n2 << " " << legendre_psi_max_size << std::endl;
  
  const size_t size = n1*n2*legendre_psi_max_size;
  checkCudaErrors(cudaMalloc(&legendre_psi_dev, size*sizeof(Complex)));
  checkCudaErrors(cudaMemset(legendre_psi_dev, 0, size*sizeof(Complex)));
}

void OmegaWaveFunction::forward_legendre_transform()
{
  setup_legendre_transform();
  
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int &n_theta = theta->n;
  const int n_legs = lmax - omega + 1;
  
  const Complex one(1.0, 0.0);
  const Complex zero(0.0, 0.0);

  insist(cublasZgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     n1*n2, n_legs, n_theta, 
                     (const cuDoubleComplex *) &one,
                     (const cuDoubleComplex *) psi_dev, n1*n2,
                     (const cuDoubleComplex *) weighted_associated_legendres_dev, n_theta,
                     (const cuDoubleComplex *) &zero,
                     (cuDoubleComplex *) legendre_psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWaveFunction::backward_legendre_transform()
{
  setup_legendre_transform();
  
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int &n_theta = theta->n;
  const int n_legs = lmax - omega + 1;
  
  const Complex one(1.0, 0.0);
  const Complex zero(0.0, 0.0);

  insist(cublasZgemm(*cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     n1*n2, n_theta, n_legs,
                     (const cuDoubleComplex *) &one,
                     (const cuDoubleComplex *) legendre_psi_dev, n1*n2,
                     (const cuDoubleComplex *) associated_legendres_dev, n_legs,
                     (const cuDoubleComplex *) &zero,
                     (cuDoubleComplex *) psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWaveFunction::forward_fft_for_legendre_psi()
{ 
  insist(cufft_plan_for_legendre_psi && legendre_psi_dev);
  
  insist(cufftExecZ2Z(*cufft_plan_for_legendre_psi, (cuDoubleComplex *) legendre_psi_dev,
		      (cuDoubleComplex *) legendre_psi_dev, CUFFT_FORWARD) == CUFFT_SUCCESS);
}

void OmegaWaveFunction::backward_fft_for_legendre_psi(const int do_scale)
{
  insist(cufft_plan_for_legendre_psi && legendre_psi_dev);
  
  insist(cufftExecZ2Z(*cufft_plan_for_legendre_psi, (cuDoubleComplex *) legendre_psi_dev, 
		      (cuDoubleComplex *) legendre_psi_dev, CUFFT_INVERSE) == CUFFT_SUCCESS);
  
  if(do_scale) {
    insist(cublas_handle);
    const int &n1 = r1->n;
    const int &n2 = r2->n;
    const int n_legs = lmax - omega + 1;
    const double s = 1.0/(n1*n2);
    insist(cublasZdscal(*cublas_handle, n1*n2*n_legs, &s, (cuDoubleComplex *) legendre_psi_dev, 1) 
           == CUBLAS_STATUS_SUCCESS);
  }
}

double OmegaWaveFunction::module_for_legendre_psi() const
{
  insist(legendre_psi_dev);

  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int n_legs = lmax - omega + 1;
  
  const cuDoubleComplex *legendre_psi_dev_ = (cuDoubleComplex *) legendre_psi_dev;

  double sum= 0.0;
  for(int l = 0; l < n_legs; l++) {
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(*cublas_handle, n1*n2, legendre_psi_dev_, 1, legendre_psi_dev_, 1, 
		       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    sum += dot.real();
    legendre_psi_dev_ += n1*n2;
  }
  
  sum *= r1->dr*r2->dr;
  return sum;
}

double OmegaWaveFunction::kinetic_energy_for_legendre_psi(const int do_fft)
{
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  const int n_legs = lmax - omega + 1;
  
  if(do_fft) forward_fft_for_legendre_psi();
  
  insist(work_dev);
  cuDoubleComplex *psi_tmp_dev = (cuDoubleComplex *) work_dev;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2);

  const cuDoubleComplex *legendre_psi_dev_ = (cuDoubleComplex *) legendre_psi_dev;
  
  double sum = 0.0;
  for(int l = 0; l < n_legs; l++) {
    
    _psi_times_kinitic_energy_<<<n_blocks, n_threads, (n1+n2)*sizeof(double)>>>
      ((Complex *) psi_tmp_dev, (const Complex *) legendre_psi_dev_, n1, n2);
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    Complex dot(0.0, 0.0);
    insist(cublasZdotc(*cublas_handle, n1*n2, legendre_psi_dev_, 1, psi_tmp_dev, 1, 
                       (cuDoubleComplex *) &dot) == CUBLAS_STATUS_SUCCESS);
    
    legendre_psi_dev_ += n1*n2;
    
    sum += dot.real();
  }

  sum *= r1->dr*r2->dr/n1/n2;

  if(do_fft) backward_fft_for_legendre_psi(1);
  
  return sum;
}


void OmegaWaveFunction::mpi_test()
{
  
  forward_legendre_transform();
  
  insist(legendre_psi_dev);
  insist(work_dev);
  
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  //const int n_legs = lmax - omega + 1;

  int rank = -1;
  insist(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
  
  int n_procs = -100;
  insist(MPI_Comm_size(MPI_COMM_WORLD, &n_procs) == MPI_SUCCESS);
  
  Complex *p_tmp = new Complex [2*n1*n2 + 1];
  insist(p_tmp);

  int *omega_l = (int *) p_tmp;
  
  if(rank == 0) omega_l[0] = omega;

  for(int l = 0; l < 10; l++) {
    if(rank == 0) {
      omega_l[1] = l+omega;
      checkCudaErrors(cudaMemcpy(p_tmp+1, legendre_psi_dev+l*n1*n2, n1*n2*sizeof(Complex),
				 cudaMemcpyDeviceToHost));
    }
    
    insist(MPI_Bcast(p_tmp, 2*(n1*n2+1), MPI_DOUBLE, 0, MPI_COMM_WORLD) == MPI_SUCCESS);
    
    //checkCudaErrors(cudaMemcpy(work_dev,
    
  }
  
  if(p_tmp) { delete [] p_tmp; p_tmp = 0; }
}

void OmegaWaveFunction::copy_data_to(const int l, Complex *phi)
{
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  
  if(l < omega) {
    memset(phi, 0, (n1*n2+1)*sizeof(Complex));
    return;
  }
  
  insist(legendre_psi_dev);
  
  const Complex *leg_phi_dev = legendre_psi_dev + (l-omega)*n1*n2;
  
  int *phi_tmp = (int *) phi;
  phi_tmp[0] = omega;
  
  checkCudaErrors(cudaMemcpy(phi+1, leg_phi_dev, n1*n2*sizeof(Complex),
			     cudaMemcpyDeviceToHost));
  
}

void OmegaWaveFunction::copy_data_from(const int l, const Complex *phi)
{
  const int &n1 = r1->n;
  const int &n2 = r2->n;
  
  if(l < omega) return;
  
  // const int *p = (const int *) phi;
  // insist(p[0] == omega);

  checkCudaErrors(cudaMemcpy(work_dev, phi+1, n1*n2*sizeof(Complex),
			     cudaMemcpyHostToDevice));
}
