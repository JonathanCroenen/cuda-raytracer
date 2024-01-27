#pragma once

#include "hit_record.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/gpu_managed.h"
#include <curand_kernel.h>

namespace rtx {

struct HitRecord;

class Lambertian {
private:
    using vec3 = math::vec3<float>;

public:
    Lambertian(const vec3& albedo) : _albedo(albedo) {}
    Lambertian(const Lambertian& other) : _albedo(other._albedo) {}
    Lambertian(Lambertian&& other) : _albedo(other._albedo) {}

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;

private:
    vec3 _albedo;
};

} // namespace rtx
