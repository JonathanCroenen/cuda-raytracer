#pragma once

#include "primitives/disk.h"
#include "primitives/plane.h"
#include "primitives/sphere.h"
#include "primitives/triangle.h"
#include "primitives/quad.h"
#include "ray.h"
#include "utils/cuda.h"
#include "utils/variant.h"

namespace rtx {

class Volume : utils::Variant<Sphere, Plane, Triangle, Disk, Quad> {
public:
    using Variant::Variant;

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;
};

} // namespace rtx
