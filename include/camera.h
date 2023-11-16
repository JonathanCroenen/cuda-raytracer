#pragma once

#include "ray.h"
#include "math/vec3.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Camera : public utils::GPUManaged {
public:
    using vec3 = math::vec3<float>;

    Camera() {}
    Camera(const vec3& origin, const vec3& look_at, const vec3& up, float fov,
           float aspect);

    GPU_FUNC Ray generate_ray(float u, float v) const;

    CPU_GPU_FUNC vec3 origin() const { return _origin; }

private:
    vec3 _origin;
    vec3 _forward;
    vec3 _right;
    vec3 _up;
};

} // namespace rtx
