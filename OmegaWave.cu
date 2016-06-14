
#include "OmegaWave.h"

void OmegaWaveFunction::setup_data(MatlabArray<Complex> &m_psi_)
{
  m_psi = &m_psi_;
  insist(m_psi);

  psi = m_psi->data;
  insist(psi);

  setup_device_data();
}

OmegaWaveFunction::~OmegaWaveFunction()
{
  m_psi = 0; 
  psi = 0;
  _CUDA_FREE_(psi_dev);
}

void OmegaWaveFunction::setup_device_data()
{
  if(psi_dev) return;

  insist(m_psi);
  
  const int &n_dims = m_psi->n_dims();
  insist(n_dims == 3);
  
  const size_t *dims = m_psi->dims();
  
  const int n1 = dims[0]/2;
  const int &n2 = dims[1];
  const int &n_theta = dims[2];

  insist(2*n1 == dims[0]);
  
  const int n = n1*n2*n_theta;

  std::cout << " size " << n1 << " " << n2 << " " << n_theta << std::endl;
  
  if(!psi_dev) {
    checkCudaErrors(cudaMalloc(&psi_dev, n*sizeof(Complex)));
    checkCudaErrors(cudaMemcpy(psi_dev, psi, n*sizeof(Complex), cudaMemcpyHostToDevice));
  }
}
