#include "primitives/sphere.h"

namespace rtx {

GPU_FUNC bool Sphere::intersect(const Ray& ray, float t_min, float t_max,
                                HitRecord& record) const {
    vec3 oc = ray.origin - _center;
    float a = vec3::dot(ray.direction, ray.direction);
    float b = vec3::dot(oc, ray.direction);
    float c = vec3::dot(oc, oc) - _radius * _radius;
    float discriminant = b * b - a * c;

    if (discriminant > 0) {
        float dist = (-b - sqrt(discriminant)) / a;
        if (dist < t_max && dist > t_min) {
            record.t = dist;
            record.pos = ray.point_at(dist);
            record.normal = (record.pos - _center) / _radius;
            record.material = &_material;

            return true;
        }

        dist = (-b + sqrt(discriminant)) / a;
        if (dist < t_max && dist > t_min) {
            record.t = dist;
            record.pos = ray.point_at(dist);
            record.normal = (record.pos - _center) / _radius;
            record.material = &_material;

            return true;
        }
    }

    return false;
}

} // namespace rtx
