#pragma once

#include "math/vec3.h"
#include "hit_record.h"
#include "ray.h"
#include "materials/material.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Sphere : public utils::GPUManaged {
private:
    using vec3 = math::vec3<float>;

public:
    Sphere() {}
    Sphere(const vec3& center, float radius, const Material& material)
        : _center(center), _radius(radius), _material(material) {}

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    vec3 _center;
    float _radius;
    Material _material;
};

} // namespace rtx
