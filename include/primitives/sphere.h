#pragma once

#include "math/vec3.h"
#include "hit_record.h"
#include "ray.h"
#include "utils/gpu_allocated.h"

namespace rtx {

class Sphere : public GPUAllocated {
public:
    Sphere() {}
    Sphere(const vec3<float>& center, float radius, const vec3<float>& color)
        : _center(center), _radius(radius), _color(color) {}

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    using vec3 = vec3<float>;

    vec3 _center;
    float _radius;
    vec3 _color;
};

} // namespace rtx
