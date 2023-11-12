#pragma once

#include "ray.h"
#include "vec3.h"
#include "utils.h"

namespace rtx {

class Camera : public GPUAllocated {
public:
    using vec3 = vec3<float>;

    Camera() {}
    Camera(const vec3& origin,
           const vec3& look_at,
           const vec3& up,
           float fov,
           float aspect) {
        float theta = fov * 3.141592f / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;

        _origin = origin;
        _forward = vec3::normalized(look_at - origin);
        _right = half_width * vec3::normalized(vec3::cross(_forward, up));
        _up = half_height * vec3::normalized(vec3::cross(_right, _forward));

    }

    __device__ Ray generate_ray(float u, float v) const {
        u = -1.0f + 2.0f * u;
        v = -1.0f + 2.0f * v;

        return Ray(_origin, _forward + u * _right + v * _up);
    }

private:
    vec3 _origin;
    vec3 _forward;
    vec3 _right;
    vec3 _up;
};

} // namespace rtx
