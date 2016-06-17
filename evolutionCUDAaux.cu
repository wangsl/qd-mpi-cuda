
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
  for(int i = 0; i < n; i++)
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
  
  for(int i = 0; i < r.n; i++)
    printf(" %.15f %.15f\n", r.left+i*r.dr, dump[i]);
}
