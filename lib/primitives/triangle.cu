#include "primitives/triangle.h"

namespace rtx {

GPU_FUNC bool Triangle::intersect(
    const Ray& ray, float t_min, float t_max, HitRecord& record) const {
  const float EPSILON = 0.0000001f;

  vec3 edge1 = _v1 - _v0;
  vec3 edge2 = _v2 - _v0;
  vec3 h = vec3::cross(ray.direction, edge2);
  float a = vec3::dot(edge1, h);
  if (a > -EPSILON && a < EPSILON) {
    return false;
  }

  float f = 1.0f / a;
  vec3 s = ray.origin - _v0;
  float u = f * vec3::dot(s, h);
  if (u < 0.0f || u > 1.0f) {
    return false;
  }

  vec3 q = vec3::cross(s, edge1);
  float v = f * vec3::dot(ray.direction, q);
  if (v < 0.0f || u + v > 1.0f) {
    return false;
  }

  float t = f * vec3::dot(edge2, q);
  if (t < t_min || t > t_max) {
    return false;
  }

  record.t = t;
  record.pos = ray.point_at(t);
  record.normal = _normal;

  return true;
}

} // namespace rtx
