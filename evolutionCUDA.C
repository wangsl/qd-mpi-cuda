
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
  pot_dev(0)
{
  allocate_device_data();
  setup_Omega_Psis();
  cudaUtils::gpu_memory_usage();
}

EvolutionCUDA::~EvolutionCUDA()
{
  pot = 0;
  deallocate_device_data();
  //cudaUtils::gpu_memory_usage();
}
  
void EvolutionCUDA::test()
{
  // std::cout << " Data Test " << std::endl;

  // std::cout << " " << r1.n << " " << r1.left << " " << r1.dr << " " << r1.mass << std::endl;
  // std::cout << " " << r2.n << " " << r2.left << " " << r2.dr << " " << r2.mass << std::endl;

  //test_device();

  //cudaUtils::gpu_memory_usage();
}
