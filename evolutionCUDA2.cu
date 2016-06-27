
#include <mpi.h>

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
    const int &n1 = r1.n;
    const int &n2 = r2.n; 
    const int &n_theta = theta.n;
    const size_t size = n1*n2*n_theta;
    std::cout << " Allocate device memory for potential: " << n1 << " " << n2 << " " << n_theta << std::endl;
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
  destroy_cufft_plan_for_psi();
  destroy_cufft_plan_for_legendre_psi();
  destroy_cublas_handle();
  _CUDA_FREE_(pot_dev);
}

void EvolutionCUDA::setup_omega_psis()
{
  const Vec<int> omgs = omegas.omegas;
  const Vec<RMat> &ass_legendres = omegas.associated_legendres;	
  Vec<Vec<Complex> > &wave_packets = omegas.wave_packets;
  
  const int n_omgs = omgs.size();
  
  omega_psis.resize(n_omgs);
  
  for (int i = 0; i < n_omgs; i++)
    omega_psis[i].setup_data(omgs[i], omegas.lmax, legendre_psi_max_size,
			     ass_legendres[i], 
			     (Complex *) wave_packets[i], pot_dev,
			     &r1, &r2, &theta,
			     0, &cufft_plan_for_legendre_psi(), 
			     &cublas_handle());
  
}

void EvolutionCUDA::test_device()
{
  double *dump1 = 0;
  checkCudaErrors(cudaMalloc(&dump1, r1.n*sizeof(double)));
  int n_threads = _NTHREADS_;
  int n_blocks = cudaUtils::number_of_blocks(n_threads, r1.n);
  _calculate_dump_function_<<<n_blocks, n_threads>>>(dump1, 1);
  _show_dump_function_<<<1,1>>>(dump1, 1);
  _CUDA_FREE_(dump1);
  
  double *dump2 = 0;
  checkCudaErrors(cudaMalloc(&dump2, r2.n*sizeof(double)));
  n_threads = _NTHREADS_;
  n_blocks = cudaUtils::number_of_blocks(n_threads, r2.n);
  _calculate_dump_function_<<<n_blocks, n_threads>>>(dump2, 2);
  _show_dump_function_<<<1,1>>>(dump2, 2);
  _CUDA_FREE_(dump2);
}

void EvolutionCUDA::setup_cufft_plan_for_psi()
{
  if(has_cufft_plan_for_psi) return;
  
  const int n1 = r1.n;
  const int n2 = r2.n;
  const int n_theta = theta.n;
  const int dim [] = { n2, n1 };
  
  insist(cufftPlanMany(&_cufft_plan_for_psi, 2, const_cast<int *>(dim), NULL, 1, n1*n2, NULL, 1, n1*n2,
		       CUFFT_Z2Z, n_theta) == CUFFT_SUCCESS);

  has_cufft_plan_for_psi = 1;
}

void EvolutionCUDA::destroy_cufft_plan_for_psi()
{
  if(!has_cufft_plan_for_psi) return;
  insist(cufftDestroy(_cufft_plan_for_psi) == CUFFT_SUCCESS);
  has_cufft_plan_for_psi = 0;
}

cufftHandle &EvolutionCUDA::cufft_plan_for_psi()
{
  setup_cufft_plan_for_psi();
  return _cufft_plan_for_psi;
}

void EvolutionCUDA::setup_cublas_handle()
{
  if(has_cublas_handle) return;
  insist(cublasCreate(&_cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 1;
}

void EvolutionCUDA::destroy_cublas_handle()
{
  if(!has_cublas_handle) return;
  insist(cublasDestroy(_cublas_handle) == CUBLAS_STATUS_SUCCESS);
  has_cublas_handle = 0;
}

cublasHandle_t &EvolutionCUDA::cublas_handle()
{
  setup_cublas_handle();
  return _cublas_handle;
}

void EvolutionCUDA::evolution_with_potential(const double dt)
{
  for (int i = 0; i < omega_psis.size(); i++) 
    omega_psis[i].evolution_with_potential(pot_dev, dt);
}

void EvolutionCUDA::setup_cufft_plan_for_legendre_psi()
{
  if(has_cufft_plan_for_legendre_psi) return;
  
  const int n1 = r1.n;
  const int n2 = r2.n;
  const int &n_legs = legendre_psi_max_size;
  
  const int dim [] = { n2, n1 };
  
  insist(cufftPlanMany(&_cufft_plan_for_legendre_psi, 2, const_cast<int *>(dim), NULL, 1,
		       n1*n2, NULL, 1, n1*n2,
		       CUFFT_Z2Z, n_legs) == CUFFT_SUCCESS);

  has_cufft_plan_for_legendre_psi = 1;
}

void EvolutionCUDA::destroy_cufft_plan_for_legendre_psi()
{
  if(!has_cufft_plan_for_legendre_psi) return;
  insist(cufftDestroy(_cufft_plan_for_legendre_psi) == CUFFT_SUCCESS);
  has_cufft_plan_for_legendre_psi = 0;
}

cufftHandle &EvolutionCUDA::cufft_plan_for_legendre_psi()
{
  setup_cufft_plan_for_legendre_psi();
  return _cufft_plan_for_legendre_psi;
}

void EvolutionCUDA::test_device_mpi()
{
  const int n_omega_psis = omega_psis.size();

  //insist(n_omega_psis == 2);

  int rank = -1;
  insist(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
  
  int n_procs = -100;
  insist(MPI_Comm_size(MPI_COMM_WORLD, &n_procs) == MPI_SUCCESS);
  
  const int &n1 = r1.n;
  const int &n2 = r2.n;

  Complex *phi_ = new Complex [n_omega_psis*(n1*n2+1)];
  insist(phi_);
  
  Complex *phi_all_ = new Complex [n_omega_psis*(n1*n2+1)*n_procs];
  insist(phi_all_);
  memset(phi_all_, 0, sizeof(Complex)*n_omega_psis*(n1*n2+1)*n_procs);
  
  const int l_max = omegas.lmax;

  for(int i = 0; i < n_omega_psis; i++) {
    omega_psis[i].forward_legendre_transform();
    omega_psis[i].forward_fft_for_legendre_psi();
  }

  for(int l = 0; l < l_max; l++) {

    for(int i = 0; i < n_omega_psis; i++) 
      omega_psis[i].copy_data_to(l, phi_+i*(n1*n2+1));
    
    insist(MPI_Allgather(phi_, n_omega_psis*(n1*n2+1), MPI_C_DOUBLE_COMPLEX, 
			 phi_all_, n_omega_psis*(n1*n2+1), MPI_C_DOUBLE_COMPLEX, 
			 MPI_COMM_WORLD) == MPI_SUCCESS);
    
    for(int j = 0; j < n_procs; j++) {
      const Complex *p_ = phi_all_ + j*n_omega_psis*(n1*n2+1);
      for(int i = 0; i < n_omega_psis; i++) {
	p_ += i*(n1*n2+1);
	omega_psis[i].copy_data_from(l, p_);
      }
    }
  }
  
  for(int i = 0; i < n_omega_psis; i++) {
    omega_psis[i].backward_fft_for_legendre_psi(1);
    omega_psis[i].backward_legendre_transform();
  }
  
  if(phi_) { delete [] phi_; phi_ = 0; }
  if(phi_all_) { delete [] phi_all_; phi_all_ = 0; }
}
