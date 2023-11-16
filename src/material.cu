#include "materials/material.h"

namespace rtx {

Material::Material(const Material& material) {
    _type = material._type;
    switch (_type) {
    case Type::LAMBERTIAN:
        _data.lambertian = material._data.lambertian;
        break;
    case Type::METAL:
        _data.metal = material._data.metal;
        break;
    case Type::DIELECTRIC:
        _data.dielectric = material._data.dielectric;
        break;
    case Type::EMISSIVE:
        _data.emissive = material._data.emissive;
        break;
    }
}


Material::~Material() {
    switch (_type) {
    case Type::LAMBERTIAN:
        _data.lambertian.~Lambertian();
        break;
    case Type::METAL:
        _data.metal.~Metal();
        break;
    case Type::DIELECTRIC:
        _data.dielectric.~Dielectric();
        break;
    case Type::EMISSIVE:
        _data.emissive.~Emissive();
        break;
    }
}

GPU_FUNC bool Material::scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                          Ray& scattered, curandState* rand_state) const {
    switch (_type) {
    case Type::LAMBERTIAN:
        return _data.lambertian.scatter(ray, record, attenuation, scattered, rand_state);
    case Type::METAL:
        return _data.metal.scatter(ray, record, attenuation, scattered, rand_state);
    case Type::DIELECTRIC:
        return _data.dielectric.scatter(ray, record, attenuation, scattered, rand_state);
    case Type::EMISSIVE:
        return _data.emissive.scatter(ray, record, attenuation, scattered, rand_state);
    }

    return false;
}

} // namespace rtx
