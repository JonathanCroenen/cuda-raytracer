#pragma once

#include "math/vec3.h"
#include "ray.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"
#include <memory>

namespace rtx {

class Camera : public utils::GPUManaged {
private:
    using vec3 = math::vec3<float>;

public:
    static std::unique_ptr<Camera> create(const vec3& origin, const vec3& look_at,
                                          const vec3& up, float fov, float aspect);

    GPU_FUNC Ray generate_ray(float u, float v) const;

private:
    Camera() = default;
    Camera(const vec3& origin, const vec3& look_at, const vec3& up, float fov,
           float aspect);

private:
    vec3 _origin;
    vec3 _forward;
    vec3 _right;
    vec3 _up;
};

} // namespace rtx
