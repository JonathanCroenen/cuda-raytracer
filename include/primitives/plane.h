#pragma once

#include "hit_record.h"
#include "materials/material.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Plane : public utils::GPUManaged {
private:
    using vec3 = math::vec3<float>;

public:
    Plane() {}
    Plane(const vec3& position, const vec3& normal, const Material& material)
        : _position(position), _normal(normal), _material(material) {}

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    vec3 _position;
    vec3 _normal;
    Material _material;
};

} // namespace rtx
