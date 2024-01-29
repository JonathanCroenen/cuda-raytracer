#include "materials/emissive.h"

namespace rtx {

GPU_FUNC bool Emissive::scatter(const Ray& ray,
                                const HitRecord& record,
                                vec3& attenuation,
                                Ray& scattered,
                                curandState* rand_state) const {
  attenuation = _albedo;
  return false;
}

} // namespace rtx
