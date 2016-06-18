
#include <mpi.h>
#include "evolutionCUDA.h"

EvolutionCUDA::EvolutionCUDA(const double *pot,
			     const RadialCoordinate &r1,
			     const RadialCoordinate &r2,
			     const AngleCoordinate &theta,
			     OmegaStates &omegas
			     ) :
  pot(pot),
  r1(r1), r2(r2), theta(theta), omegas(omegas),
  has_setup_constant_memory(0),
  has_cufft_plan_for_psi(0),
  has_cublas_handle(0),
  pot_dev(0)
{
  allocate_device_data();
  setup_omega_psis();
  cudaUtils::gpu_memory_usage();
}

EvolutionCUDA::~EvolutionCUDA()
{
  pot = 0;
  deallocate_device_data();
}
  
void EvolutionCUDA::test()
{
  std::cout << " ";
  for (int k = 1; k <= 10000; k++) {
    if (k%100 == 0) { 
      std::cout << k << " ";
      if (k%1000 == 0) std::cout << std::endl << " ";
    }
    for (int i = 0; i < omega_psis.size(); i++) {
      //Omega_Psis[i].test();
      omega_psis[i].test(cufft_plan_for_psi(), cublas_handle());
    }
    insist(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  }
  
  std::cout << std::endl;
}
