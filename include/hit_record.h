#pragma once

#include "math/vec3.h"

namespace rtx {

struct HitRecord {
    float t;
    vec3<float> pos;
    vec3<float> normal;
    vec3<float> color;
};

} // namespace rtx
