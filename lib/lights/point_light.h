#pragma once

#include "lights/light_sample.h"
#include "math/vec3.h"

namespace rtx {

class PointLight {
private:
  using vec3 = math::vec3<float>;

public:
  PointLight() = default;
  PointLight(const vec3& position, const vec3& color, float intensity)
      : position(position), color(color), intensity(intensity) {}

  GPU_FUNC const LightSample sample(const vec3& from, curandState* rand_state) const;

  vec3 position;
  vec3 color;
  float intensity;
};

} // namespace rtx
