#pragma once

#include "math/vec3.h"
#include "utils/cuda.h"
#include "utils/gpu_allocated.h"
#include "ray.h"
#include "hit_record.h"

namespace rtx {

class Plane : public GPUAllocated {
public:
    Plane() {}
    Plane(const vec3<float>& position, const vec3<float>& normal, const vec3<float>& color)
        : _position(position), _normal(normal), _color(color) {}

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    using vec3 = vec3<float>;

    vec3 _position;
    vec3 _normal;
    vec3 _color;
};

} // namespace rtx
