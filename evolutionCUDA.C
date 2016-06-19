
#include <mpi.h>
#include "evolutionCUDA.h"

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
  std::cout << " time_step: " << time.total_steps << " steps: " << time.steps 
	    << " time_step: " << time.time_step << std::endl;
  
  for (int k = 1; k <= 2; k++) {
    time.steps++;
    for (int i = 0; i < omega_psis.size(); i++) {
      std::cout << " module: " << omega_psis[i].module() << std::endl;
    }
    insist(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  }
}
