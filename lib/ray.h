#pragma once

#include "math/vec3.h"
#include "utils/cuda.h"

namespace rtx {

struct Ray {
private:
  using vec3 = math::vec3<float>;

public:
  GPU_FUNC Ray(){};
  GPU_FUNC Ray(const vec3& origin, const vec3& direction)
      : origin(origin), direction(std::move(vec3::normalized(direction))) {}

  GPU_FUNC vec3 point_at(float t) const { return origin + t * direction; }

  vec3 origin;
  vec3 direction;
};

} // namespace rtx
