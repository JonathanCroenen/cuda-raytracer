#include "lights/light.h"

namespace rtx {

GPU_FUNC const LightSample Light::sample(const vec3& from, curandState* rand_state) const {
  switch (type_id()) {
  case type_id_of<PointLight>(): return get<PointLight>().sample(from, rand_state);
  default: return LightSample(); // Unreachable
  }
}

} // namespace rtx
