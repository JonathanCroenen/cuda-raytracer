#include "primitives/disk.h"

namespace rtx {

GPU_FUNC bool Disk::intersect(
    const Ray& ray, float t_min, float t_max, HitRecord& record) const {
  float denom = vec3::dot(ray.direction, _normal);
  if (denom > 0.0f) {
    return false;
  }

  float dist = vec3::dot(_center - ray.origin, _normal) / denom;
  if (dist < t_min || dist > t_max) {
    return false;
  }

  vec3 pos = ray.point_at(dist);
  vec3 d = pos - _center;
  if (d.squared_length() > _radius * _radius) {
    return false;
  }

  record.t = dist;
  record.pos = pos;
  record.normal = _normal;

  return true;
}

} // namespace rtx
