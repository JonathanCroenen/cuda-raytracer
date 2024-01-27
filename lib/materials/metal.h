#pragma once

#include "hit_record.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/gpu_managed.h"
#include <curand_kernel.h>

namespace rtx {

struct HitRecord;

class Metal {
private:
    using vec3 = math::vec3<float>;

public:
    Metal(const vec3& albedo, float fuzz) : _albedo(albedo), _fuzz(fuzz) {}
    Metal(const Metal& other) : _albedo(other._albedo), _fuzz(other._fuzz) {}
    Metal(Metal&& other) : _albedo(other._albedo), _fuzz(other._fuzz) {}

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;

private:
    vec3 _albedo;
    float _fuzz;
};

} // namespace rtx
