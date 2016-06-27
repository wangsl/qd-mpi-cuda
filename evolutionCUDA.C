
#include <climits>
#include <mpi.h>
#include "evolutionCUDA.h"

#include <chrono>
#include <ctime>

inline char *time_now()
{
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  std::time_t time_now  = std::chrono::system_clock::to_time_t(now);
  return std::ctime(&time_now); 
}

EvolutionCUDA::EvolutionCUDA(const double *pot_,
			     const RadialCoordinate &r1_,
			     const RadialCoordinate &r2_,
			     const AngleCoordinate &theta_,
			     OmegaStates &omegas_,
			     EvolutionTime &time_
			     ) :
  pot(pot_),
  r1(r1_), r2(r2_), theta(theta_), 
  omegas(omegas_), time(time_),
  legendre_psi_max_size(0),
  has_setup_constant_memory(0),
  has_cufft_plan_for_psi(0),
  has_cufft_plan_for_legendre_psi(0),
  has_cublas_handle(0),
  pot_dev(0)
{
  setup_legendre_psi_max_size();
  allocate_device_data();
  setup_omega_psis();
  cudaUtils::gpu_memory_usage();
}

EvolutionCUDA::~EvolutionCUDA()
{
  pot = 0;
  deallocate_device_data();
}

void EvolutionCUDA::setup_legendre_psi_max_size()
{
  const int l_max = omegas.lmax;
  int omega_min = INT_MAX;
  for(int i = 0; i < omegas.omegas.size(); i++) {
    if(omegas.omegas[i] < omega_min)
      omega_min = omegas.omegas[i];
  }
  legendre_psi_max_size = l_max - omega_min + 1;
  std::cout << " legendre_psi_max_size: " << legendre_psi_max_size << std::endl;
}
  
void EvolutionCUDA::test()
{
  std::cout << " time_step: " << time.total_steps << " steps: " << time.steps 
	    << " time_step: " << time.time_step << std::endl;

  std::chrono::time_point<std::chrono::system_clock> now;

  for(int i = 0; i < time.total_steps; i++) {
    time.steps++;
    std::cout << " steps: " << time.steps << " " << time_now();
    test_device_mpi();
  }

  std::cout << " time_step: " << time.total_steps << " steps: " << time.steps 
	    << " time_step: " << time.time_step << std::endl;
  
  return;

  for(int i = 0; i < omega_psis.size(); i++) {
    omega_psis[i].mpi_test();
  }
  
 return;
 
 for (int k = 0; k < 2; k++) {
   time.steps++;
   for (int i = 0; i < omega_psis.size(); i++) {
     std::cout << " psi module: " << omega_psis[i].module_for_psi() << std::endl;
     std::cout << " potential energy: " << omega_psis[i].potential_energy() << std::endl;
     omega_psis[i].test();
     std::cout << " legendre psi module: " << omega_psis[i].module_for_legendre_psi() << std::endl;
   }
   insist(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  }
}
