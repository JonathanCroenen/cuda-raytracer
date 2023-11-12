#pragma once

#include "vec3.h"

namespace rtx {

struct Ray {
    using vec3 = vec3<float>;

    __device__ Ray(){};
    __device__ Ray(const vec3& origin, const vec3& direction)
        : origin(origin), direction(std::move(vec3::unit_vector(direction))) {}

    __device__ vec3 point_at(float t) const { return origin + t * direction; }

    const vec3 origin;
    const vec3 direction;
};

} // namespace rtx
