#include "materials/lambertian.h"

namespace rtx {

GPU_FUNC bool Lambertian::scatter(const Ray& ray, const HitRecord& record,
                                  vec3& attenuation, Ray& scattered,
                                  curandState* rand_state) const {
    vec3 target = record.pos + record.normal + vec3::random_unit(rand_state);
    scattered = Ray(record.pos, target - record.pos);
    attenuation = _albedo;

    return true;
}

}
