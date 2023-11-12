#pragma once

#include "sphere.h"

#include <thrust/device_vector.h>

namespace rtx {

class Scene : public GPUAllocated {
public:
    Scene() {}

    void add(const Sphere& sphere) {
        _spheres.push_back(sphere);
    }

    __device__ bool hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const {
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (const Sphere& sphere : _spheres) {
            if (sphere.hit(ray, t_min, closest_so_far, record)) {
                hit_anything = true;
                closest_so_far = record.t;
            }
        }

        return hit_anything;
    }

private:

    thrust::device_vector<Sphere> _spheres;
};

} // namespace rtx
