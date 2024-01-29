#include "lights/light.h"

namespace rtx {

GPU_FUNC const LightSample Light::sample(const vec3& from, curandState* rand_state) const {
  if (is<PointLight>()) {
    get<PointLight>().sample(from, rand_state);
  }
}

} // namespace rtx
