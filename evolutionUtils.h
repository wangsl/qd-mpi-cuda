
#ifndef EVOLUTION_UTILS_H
#define EVOLUTION_UTILS_H

#include <helper_cuda.h>

#ifdef printf
#undef printf
#endif

#ifdef __NVCC__

namespace EvoltionUtils {

  struct RadialCoordinate
  {
    double left;
    double dr;
    double mass;
    double dump_Cd;
    double dump_xd;
    int n;
  };

  inline void copy_radial_coordinate_to_device(const RadialCoordinate &r_dev, 
					       const double &left, const double &dr,
					       const double &mass, 
					       const double &dump_Cd, const double &dump_xd,
					       const int &n)
  {
    RadialCoordinate r;
    r.left = left;
    r.dr = dr;
    r.mass = mass;
    r.dump_Cd = dump_Cd;
    r.dump_xd = dump_xd;
    r.n = n;
    checkCudaErrors(cudaMemcpyToSymbol(r_dev, &r, sizeof(RadialCoordinate)));
  }
}

// These constant memory variables are defined as evolutionCUDA2.cu

extern __constant__ EvoltionUtils::RadialCoordinate r1_dev;
extern __constant__ EvoltionUtils::RadialCoordinate r2_dev;
extern __constant__ double gauss_legendre_weight_dev[512];

#endif /* __NVCC__ */

void setup_coriolis_matrix(const int J, const int p, const int j);

#endif /* EVOLUTION_UTILS_H */

