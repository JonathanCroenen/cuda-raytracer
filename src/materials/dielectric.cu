#include "materials/dielectric.h"

namespace rtx {

GPU_FUNC bool Dielectric::scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation,
                                  Ray& scattered, curandState* rand_state) const {
    vec3 outward_normal;
    vec3 reflected = vec3::reflect(ray.direction, rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;
    float reflect_prob;
    float cosine;

    if (vec3::dot(ray.direction, rec.normal) > 0) {
        outward_normal = -rec.normal;
        ni_over_nt = _refraction_index;
        cosine = _refraction_index * vec3::dot(ray.direction, rec.normal) /
                 ray.direction.length();
    } else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0 / _refraction_index;
        cosine = -vec3::dot(ray.direction, rec.normal) / ray.direction.length();
    }

    if (refract(ray.direction, outward_normal, ni_over_nt, refracted)) {
        reflect_prob = schlick(cosine, _refraction_index);
    } else {
        scattered = Ray(rec.pos, reflected);
        reflect_prob = 1.0;
    }

    if (curand_uniform(rand_state) < reflect_prob) {
        scattered = Ray(rec.pos, reflected);
    } else {
        scattered = Ray(rec.pos, refracted);
    }

    return true;
}

GPU_FUNC bool Dielectric::refract(const vec3& v, const vec3& n, float ni_over_nt,
                                  vec3& refracted) const {
    vec3 uv = vec3::normalized(v);
    float dt = vec3::dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);

    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }

    return false;
}

GPU_FUNC float Dielectric::schlick(float cosine, float _refraction_index) const {
    float r0 = (1 - _refraction_index) / (1 + _refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

} // namespace rtx
