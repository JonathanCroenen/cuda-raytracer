#pragma once

#include "math/vec3.h"
#include "ray.h"
// #include "utils.h"

namespace rtx {

struct Light {
    using vec3 = vec3<float>;

    Light() {}
    Light(const vec3& position, const vec3& color, float intensity)
        : position(position), color(color), intensity(intensity) {}

    vec3 position;
    vec3 color;
    float intensity;
};

} // namespace rtx
