#pragma once

#include "vec3.h"
#include "sphere.h"
#include "light.h"
#include "camera.h"

#include <thrust/device_vector.h>

namespace rtx {

class Scene : public GPUAllocated {
public:
    using vec3 = vec3<float>;

    Scene() {}

    void add(const Sphere& sphere) { _spheres.push_back(sphere); }
    void add(const Light& light) { _lights.push_back(light); }

    __device__ bool
    hit(const Ray& ray, float t_min, float t_max, HitRecord& record) const {
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

    __device__ vec3 calc_light(const HitRecord& rec, const Camera& camera) const {
        vec3 color(0.0f);

        for (const Light& light : _lights) {
            vec3 light_direction = vec3::normalized(light.position - rec.pos);
            Ray shadow_ray(rec.pos, light_direction);

            vec3 intensity = max(0.0f, vec3::dot(rec.normal, light_direction));
            intensity *= 0.9f / 3.141592f * light.intensity;
            intensity *= rec.color * light.color;

            vec3 reflected = vec3::reflect(-light_direction, rec.normal);
            vec3 cam_direction = vec3::normalized(camera.origin() - rec.pos);
            float c = max(0.0f, vec3::dot(reflected, cam_direction));
            intensity += 0.2f * pow(c, 20.0f) * light.intensity * light.color;
            color += intensity;
        }

        return color;
    }

    __device__ thrust::device_vector<Light> get_lights() const { return _lights; }

private:
    thrust::device_vector<Sphere> _spheres;
    thrust::device_vector<Light> _lights;
};

} // namespace rtx
