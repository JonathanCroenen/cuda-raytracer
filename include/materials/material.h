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

// class Material {
// private:
//     using vec3 = math::vec3<float>;
//
// public:
//     Material() {}
//     Material(const Material& material);
//     Material(Material&& material);
//     Material(const Lambertian& lambertian) : _type(Type::LAMBERTIAN) {
//         new (&_data.lambertian) Lambertian(lambertian);
//     }
//     Material(const Metal& metal) : _type(Type::METAL) {
//         new (&_data.metal) Metal(metal);
//     }
//     Material(const Dielectric& dielectric) : _type(Type::DIELECTRIC) {
//         new (&_data.dielectric) Dielectric(dielectric);
//     }
//     Material(const Emissive& emissive) : _type(Type::EMISSIVE) {
//         new (&_data.emissive) Emissive(emissive);
//     }
//
//     ~Material();
//
//     GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
//                           Ray& scattered, curandState* rand_state) const;
//
// private:
//     enum class Type { LAMBERTIAN, METAL, DIELECTRIC, EMISSIVE } _type;
//
//     union Data {
//         Data() {}
//         ~Data() {}
//
//         Lambertian lambertian;
//         Metal metal;
//         Dielectric dielectric;
//         Emissive emissive;
//     } _data;
// };
//
// template <typename T, typename... Args>
// Material make_material(Args... args) {
//     return Material(T(args...));
// }

class Material : utils::Variant<Lambertian, Metal, Dielectric, Emissive> {
private:
    using vec3 = math::vec3<float>;

public:
    using Variant::Variant;
    using Variant::operator=;
    using Variant::is;
    using Variant::get;

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;
};

} // namespace rtx
