#pragma once

#include "math/vec3.h"
#include "hit_record.h"
#include "ray.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Sphere {
private:
    using vec3 = math::vec3<float>;

public:
    Sphere() = default;
    Sphere(const Sphere&) = default;

    Sphere(const vec3& center, float radius)
        : _center(center), _radius(radius) {}

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    vec3 _center;
    float _radius;
};

} // namespace rtx
