#pragma once

#include "vec3.h"
#include "ray.h"
#include "hit_record.h"
#include "utils.h"

namespace rtx {

class Material : public GPUAllocated {
public:
    using vec3 = vec3<float>;

    __device__ virtual bool scatter(const Ray& ray_in,
                                    const HitRecord& record,
                                    vec3& attenuation,
                                    Ray& scattered) const = 0;
};

} // namespace rtx
