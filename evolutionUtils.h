
#ifndef EVOLUTION_UTILS_H
#define EVOLUTION_UTILS_H

#include <helper_cuda.h>

#ifdef printf
#undef printf
#endif

namespace EvoltionUtils {

  struct RadialCoordinate
  {
    double left;
    double dr;
    double mass;
    int n;
  };

  inline void copy_radial_coordinate_to_device(const RadialCoordinate &r_dev, 
					       const double &left, const double &dr,
					       const double &mass, const int &n)
  {
    RadialCoordinate r;
    r.left = left;
    r.dr = dr;
    r.mass = mass;
    r.n = n;
    checkCudaErrors(cudaMemcpyToSymbol(r_dev, &r, sizeof(RadialCoordinate)));
  }
}

#endif /* EVOLUTION_UTILS_H */

