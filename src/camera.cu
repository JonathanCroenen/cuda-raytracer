#include "camera.h"

namespace rtx {

std::unique_ptr<Camera> Camera::create(const vec3& origin, const vec3& look_at,
                                       const vec3& up, float fov, float aspect) {
    return std::unique_ptr<Camera>(new Camera(origin, look_at, up, fov, aspect));
}

Camera::Camera(const vec3& origin, const vec3& look_at, const vec3& up, float fov,
               float aspect) {
    float theta = fov * 3.141592f / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;

    _origin = origin;
    _forward = vec3::normalized(look_at - origin);
    _right = half_width * vec3::normalized(vec3::cross(_forward, up));
    _up = half_height * vec3::normalized(vec3::cross(_right, _forward));
}

GPU_FUNC Ray Camera::generate_ray(float u, float v) const {
    u = -1.0f + 2.0f * u;
    v = -1.0f + 2.0f * v;

    return Ray(_origin, _forward + u * _right + v * _up);
}

} // namespace rtx
