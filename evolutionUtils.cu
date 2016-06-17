
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
  for(int i = 0; i < n; i++)
    printf(" %d %.15f\n", i, gauss_legendre_weight_dev[i]);
}

static __global__ void _calculate_dump_function_(double *dump, const int n,
						 const double r_left, const double dr, 
						 const double Cd, const double xd)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < n) {
    const double r = r_left + index*dr;
    dump[index] = cudaMath::WoodsSaxon(r, Cd, xd);
  }
}

static __global__ void _show_dump_function_(double *dump, const int n, 
					    const double r_left, const double dr)
{
  for(int i = 0; i < n; i++) {
    const double r = r_left + i*dr;
    printf("%.15f %.15f\n", r, dump[i]);
  }
}

static __global__ void _calculate_dump_function_r1_(double *dump)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if (index < r1_dev.n) {
    const double r = r1_dev.left + index*r1_dev.dr;
    dump[index] = cudaMath::WoodsSaxon(r, r1_dev.Cd, r1_dev.xd);
  }
}
