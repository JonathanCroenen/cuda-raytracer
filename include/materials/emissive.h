#pragma once

#include "hit_record.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/gpu_managed.h"
#include <curand_kernel.h>

namespace rtx {

struct HitRecord;

class Emissive : public utils::GPUManaged {
private:
    using vec3 = math::vec3<float>;

public:
    Emissive(const vec3& albedo) : _albedo(albedo) {}
    Emissive(const Emissive& other) : _albedo(other._albedo) {}

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;

private:
    vec3 _albedo;
};

} // namespace rtx
