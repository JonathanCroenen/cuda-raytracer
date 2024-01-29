#pragma once

#include "hit_record.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/cuda.h"

namespace rtx {

class Disk {
private:
  using vec3 = math::vec3<float>;

public:
  Disk() = default;
  Disk(const Disk&) = default;

  Disk(const vec3& center, const vec3& normal, float radius)
      : _center(center), _normal(normal), _radius(radius) {}

  GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max, HitRecord& record) const;

private:
  vec3 _center;
  vec3 _normal;
  float _radius;
};

} // namespace rtx
