#include "materials/material.h"

namespace rtx {

GPU_FUNC bool Material::scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const {
    if (is<Lambertian>()) {
        return get<Lambertian>().scatter(ray, record, attenuation, scattered, rand_state);
    } else if (is<Metal>()) {
        return get<Metal>().scatter(ray, record, attenuation, scattered, rand_state);
    } else if (is<Dielectric>()) {
        return get<Dielectric>().scatter(ray, record, attenuation, scattered, rand_state);
    } else if (is<Emissive>()) {
        return get<Emissive>().scatter(ray, record, attenuation, scattered, rand_state);
    } else {
        return false;
    }
}

} // namespace rtx
