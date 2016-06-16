
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

#endif /* EVOLUTION_UTILS_H */

