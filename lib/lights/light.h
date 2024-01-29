#pragma once

#include "lights/point_light.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/variant.h"

namespace rtx {

struct Light : utils::Variant<PointLight> {
private:
  using vec3 = math::vec3<float>;

public:
  using Variant::Variant;

  GPU_FUNC const LightSample sample(const vec3& from, curandState* rand_state) const;
};

} // namespace rtx
