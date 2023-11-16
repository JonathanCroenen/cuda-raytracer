#pragma once

#include "materials/dielectric.h"
#include "materials/emissive.h"
#include "materials/lambertian.h"
#include "materials/metal.h"
#include "hit_record.h"
#include "ray.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Material {
private:
    using vec3 = math::vec3<float>;

public:
    Material() {}
    Material(const Material& material);
    Material(const Lambertian& lambertian) : _type(Type::LAMBERTIAN) {
        _data.lambertian = lambertian;
    }
    Material(const Metal& metal) : _type(Type::METAL) { _data.metal = metal; }
    Material(const Dielectric& dielectric) : _type(Type::DIELECTRIC) {
        _data.dielectric = dielectric;
    }
    Material(const Emissive& emissive) : _type(Type::EMISSIVE) {
        _data.emissive = emissive;
    }

    ~Material();

    GPU_FUNC bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const;

private:
    enum class Type { LAMBERTIAN, METAL, DIELECTRIC, EMISSIVE } _type;

    union Data {
        Data() {}
        ~Data() {}

        Lambertian lambertian;
        Metal metal;
        Dielectric dielectric;
        Emissive emissive;
    } _data;
};

template <typename T, typename... Args>
Material make_material(Args... args) {
    return Material(T(args...));
}

} // namespace rtx
