#include "materials/metal.h"

namespace rtx {

GPU_FUNC bool Metal::scatter(const Ray& ray, const HitRecord& record, vec3& attenuation,
                             Ray& scattered, curandState* rand_state) const {

    vec3 reflected = vec3::reflect(vec3::normalized(ray.direction), record.normal);
    scattered = Ray(record.pos, reflected + _fuzz * vec3::random_unit(rand_state));
    attenuation = _albedo;

    return (vec3::dot(scattered.direction, record.normal) > 0);
}

} // namespace rtx
