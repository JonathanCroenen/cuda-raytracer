#pragma once

#include "ray.h"
#include "math/vec3.h"
#include "utils/gpu_allocated.h"

namespace rtx {

class Camera : public GPUAllocated {
public:
    using vec3 = vec3<float>;

    Camera() {}
    Camera(const vec3& origin, const vec3& look_at, const vec3& up, float fov,
           float aspect);

    __device__ Ray generate_ray(float u, float v) const;

    __host__ __device__ vec3 origin() const { return _origin; }

private:
    vec3 _origin;
    vec3 _forward;
    vec3 _right;
    vec3 _up;
};

} // namespace rtx
