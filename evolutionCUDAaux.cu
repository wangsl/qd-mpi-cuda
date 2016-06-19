
#include "evolutionUtils.h"
#include "cudaMath.h"

static __global__ void _test_constant_memory_()
{
  printf(" left=%.15f, dr=%.15f, mass=%.15f, Cd=%.15f, xd=%.15f, n=%d\n", 
	 r1_dev.left, r1_dev.dr, r1_dev.mass, r1_dev.dump_Cd, r1_dev.dump_xd, r1_dev.n);

  printf(" left=%.15f, dr=%.15f, mass=%.15f, Cd=%.15f, xd=%.15f, n=%d\n", 
	 r2_dev.left, r2_dev.dr, r2_dev.mass, r2_dev.dump_Cd, r2_dev.dump_xd, r2_dev.n);
}

static __global__ void _print_gauss_legendre_weight_(const int n)
{
  printf(" Gauss-Legendre weights\n");
  for (int i = 0; i < n; i++)
    printf(" %d %.15f\n", i, gauss_legendre_weight_dev[i]);
}

static __global__ void _calculate_dump_function_(double *dump, const int r_index)
{
  EvoltionUtils::RadialCoordinate r;
  if(r_index == 1)
    r = r1_dev;
  else if(r_index == 2)
    r = r2_dev;
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < r.n) 
    dump[index] = cudaMath::WoodsSaxon(r.left+index*r.dr, r.dump_Cd, r.dump_xd);
}

static __global__ void _show_dump_function_(double *dump, const int r_index)
{
  printf(" Dump function\n");
  
  EvoltionUtils::RadialCoordinate r;
  if(r_index == 1)
    r = r1_dev;
  else if(r_index == 2)
    r = r2_dev;
  
  for (int i = 0; i < r.n; i++)
    printf(" %.15f %.15f\n", r.left+i*r.dr, dump[i]);
}

static __global__ void _evolution_with_potential_(Complex *psi, const double *pot, 
						  const int n, const double dt)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) 
    psi[index] *= exp(Complex(0.0, -dt)*pot[index]);
}

static __global__ void _evolution_with_kinetic_(Complex *psi, const int n1, const int n2, const int m, 
                                                const double dt)
{
  extern __shared__ double s_data[];
  
  double *kin1 = (double *) s_data;
  double *kin2 = (double *) &kin1[n1];
  
  cudaMath::setup_kinetic_energy_for_fft(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cudaMath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);
  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*m) {
    int i = -1; int j = -1; int k = -1;
    cudaMath::index_2_ijk(index, n1, n2, m, i, j, k);
    psi[index] *= exp(Complex(0.0, -dt)*(kin1[i]+kin2[j]));
  }
}

static __global__ void _psi_times_kinitic_energy_(Complex *psiOut, const Complex *psiIn, 
                                                  const int n1, const int n2)
{
  extern __shared__ double s_data[];

  double *kin1 = (double *) s_data;
  double *kin2 = (double *) &kin1[n1];
  
  cudaMath::setup_kinetic_energy_for_fft(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cudaMath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);
  __syncthreads();

  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2) {
    int i = -1; int j = -1;
    cudaMath::index_2_ij(index, n1, n2, i, j);
    psiOut[index] = psiIn[index]*(kin1[i] + kin2[j]);
  }
}
