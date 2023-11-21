#pragma once

#include "hit_record.h"
#include "materials/dielectric.h"
#include "materials/emissive.h"
#include "materials/lambertian.h"
#include "materials/metal.h"
#include "ray.h"
#include "utils/gpu_managed.h"
#include "utils/variant.h"

namespace rtx {

class Material : utils::Variant<Lambertian, Metal, Dielectric, Emissive> {
private:
    using vec3 = math::vec3<float>;

public:
    using Variant::Variant;

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;
};

} // namespace rtx
