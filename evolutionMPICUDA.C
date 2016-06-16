
#include "evolutionMPICUDA.h"

EvolutionMPICUDA::EvolutionMPICUDA(const MatlabArray<double> &m_pot,
				   MatlabArray<Complex> &m_psi,
				   const RadialCoordinate &r1,
				   const RadialCoordinate &r2,
				   const AngleCoordinate &theta
				   ) :
  m_pot(m_pot), m_psi(m_psi),
  r1(r1), r2(r2), theta(theta),
  pot_dev(0)
{
  pot = m_pot.data;
  insist(pot);
  
  allocate_device_data();
}

EvolutionMPICUDA::~EvolutionMPICUDA()
{
  pot = 0;
  deallocate_device_data();
  cudaUtils::gpu_memory_usage();
}
  
void EvolutionMPICUDA::test()
{
  std::cout << " Data Test " << std::endl;

  const int &n_dims = m_pot.n_dims();
  const size_t *dims = m_pot.dims();
  for(int i = 0; i < n_dims; i++)
    std::cout << " " << dims[i];
  std::cout << std::endl;

  const int &n_ds = m_psi.n_dims();
  const size_t *ds = m_psi.dims();
  for(int i = 0; i < n_ds; i++)
    std::cout << " " << ds[i];
  std::cout << std::endl;

  std::cout << " " << r1.n << " " << r1.left << " " << r1.dr << " " << r1.mass << std::endl;
  std::cout << " " << r2.n << " " << r2.left << " " << r2.dr << " " << r2.mass << std::endl;

  test_device();

  cudaUtils::gpu_memory_usage();
}
