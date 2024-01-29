#pragma once

#include "hit_record.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/cuda.h"

namespace rtx {

class Quad {
private:
  using vec3 = math::vec3<float>;

public:
  Quad() = default;
  Quad(const Quad&) = default;

  Quad(const vec3& v0, const vec3& v1, const vec3& v2) : _v0(v0), _v1(v1), _v2(v2) {
    _normal = vec3::normalized(vec3::cross(_v1 - _v0, _v2 - _v0));
  }

  GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max, HitRecord& record) const;

private:
  vec3 _v0;
  vec3 _v1;
  vec3 _v2;

  vec3 _normal;
};

} // namespace rtx
