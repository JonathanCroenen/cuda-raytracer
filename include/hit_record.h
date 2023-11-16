#pragma once

#include "math/vec3.h"

namespace rtx {

class Material;

struct HitRecord {
private:
    using vec3 = math::vec3<float>;

public:
    float t;
    vec3 pos;
    vec3 normal;
    const Material* material;
};

} // namespace rtx
