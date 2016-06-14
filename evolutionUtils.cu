
#include "evolutionUtils.h"

static __global__ void _test_constant_memory_()
{
  printf(" left=%.4f, dr=%.4f, mass=%.4f, n=%d\n", r1_dev.left, r1_dev.dr, r1_dev.mass, r1_dev.n);
  printf(" left=%.4f, dr=%.4f, mass=%.4f, n=%d\n", r2_dev.left, r2_dev.dr, r2_dev.mass, r2_dev.n);
}

static __global__ void _print_gauss_legendre_weight_(const int n)
{
  for(int i = 0; i < n; i++)
    printf(" %d %.15f\n", i, gauss_legendre_weight_dev[i]);
}
