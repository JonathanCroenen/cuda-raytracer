#pragma once

#include "primitives/plane.h"
#include "primitives/sphere.h"
#include "materials/material.h"
#include "ray.h"
#include "utils/cuda.h"
#include "utils/variant.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Volume : utils::Variant<Sphere, Plane> {
public:
    using Variant::Variant;

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;
};

} // namespace rtx
