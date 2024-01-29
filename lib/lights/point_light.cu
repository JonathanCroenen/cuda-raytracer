#include "lights/point_light.h"

namespace rtx {

GPU_FUNC const LightSample PointLight::sample(const vec3& from,
                                              curandState* rand_state) const {
  LightSample sample;
  sample.position = position;
  sample.color = color;
  sample.intensity = intensity;
  return sample;
}

} // namespace rtx
