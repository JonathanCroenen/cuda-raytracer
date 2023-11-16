#pragma once

#include "hit_record.h"
#include "math/vec3.h"
#include "ray.h"
#include "utils/gpu_managed.h"
#include <curand_kernel.h>

namespace rtx {

struct HitRecord;

class Dielectric : public utils::GPUManaged {
private:
    using vec3 = math::vec3<float>;

public:
    Dielectric(float refraction_index) : _refraction_index(refraction_index) {}
    Dielectric(const Dielectric& other) : _refraction_index(other._refraction_index) {}

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;

private:
    GPU_FUNC bool refract(const vec3& v, const vec3& n, float ni_over_nt,
                          vec3& refracted) const;

    GPU_FUNC float schlick(float cosine, float refraction_index) const;

private:
    float _refraction_index;
};

} // namespace rtx
