#include "primitives/plane.h"

namespace rtx {

GPU_FUNC bool Plane::intersect(const Ray& ray, float t_min, float t_max,
                               HitRecord& record) const {
    float denom = vec3::dot(ray.direction, _normal);
    if (denom > 0) {
        return false;
    }

    float dist = vec3::dot(_position - ray.origin, _normal) / denom;
    if (dist < t_max && dist > t_min) {
        record.t = dist;
        record.pos = ray.point_at(dist);
        record.normal = _normal;
        record.material = &_material;

        return true;
    }

    return false;
}

} // namespace rtx
