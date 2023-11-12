#pragma once

#include "vec3.h"
#include "ray.h"
#include "hit_record.h"
#include "utils.h"

namespace rtx {

class Sphere : public GPUAllocated {
public:
    Sphere() {}
    Sphere(const vec3<float>& center, float radius) : _center(center), _radius(radius) {}

    __device__ bool hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const {
        vec3 oc = ray.origin - _center;
        float a = vec3::dot(ray.direction, ray.direction);
        float b = vec3::dot(oc, ray.direction);
        float c = vec3::dot(oc, oc) - _radius * _radius;
        float discriminant = b * b - a * c;

        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                record.t = temp;
                record.pos = ray.point_at(record.t);
                record.normal = (record.pos - _center) / _radius;
                return true;
            }

            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                record.t = temp;
                record.pos = ray.point_at(record.t);
                record.normal = (record.pos - _center) / _radius;
                return true;
            }
        }

        return false;
    }

private:
    using vec3 = vec3<float>;

    vec3 _center;
    float _radius;
    vec3 _color;
};

} // namespace rtx


