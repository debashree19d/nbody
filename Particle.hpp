#ifndef _PARTICLE_HPP
#define _PARTICLE_HPP
#include <cmath>

#include "type.hpp"

struct Particle {
 public:
  Particle() : pos{}, vel{}, acc{}, mass{} {};
  RealType pos[4];
  RealType vel[4];
  RealType acc[4];
  RealType mass;
};

#endif
