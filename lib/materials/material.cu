#include "materials/material.h"

namespace rtx {

GPU_FUNC bool Material::scatter(const Ray& ray,
                                const HitRecord& record,
                                vec3& attenuation,
                                Ray& scattered,
                                curandState* rand_state) const {
  switch (type_id()) {
  case type_id_of<Lambertian>():
    return get<Lambertian>().scatter(ray, record, attenuation, scattered, rand_state);
  case type_id_of<Metal>():
    return get<Metal>().scatter(ray, record, attenuation, scattered, rand_state);
  case type_id_of<Dielectric>():
    return get<Dielectric>().scatter(ray, record, attenuation, scattered, rand_state);
  case type_id_of<Emissive>():
    return get<Emissive>().scatter(ray, record, attenuation, scattered, rand_state);
  default:
    return false;
  }
}

} // namespace rtx
