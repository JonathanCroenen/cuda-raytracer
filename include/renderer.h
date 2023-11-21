#pragma once

#include "math/vec3.h"
#include "scene.h"
#include "utils/cuda.h"
#include <curand_kernel.h>

namespace rtx {

class Renderer {
private:
    using vec3 = math::vec3<float>;

public:
private:
    KERNEL_FUNC static void init_random_state(int width, int height,
                                              curandState* random_state) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }

        int pixel_index = y * width + x;
        curand_init(1984 + pixel_index, 0, 0, &random_state[pixel_index]);
    }

    KERNEL_FUNC static void render(vec3* framebuffer, int width, int height,
                                   Camera* camera, Scene* scene, curandState* rand_state) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if (i >= width || j >= height) {
            return;
        }

        int pixel_index = j * width + i;
        curandState local_rand_state = rand_state[pixel_index];

        vec3 color = vec3(0.0f);
        for (int s = 0; s < 2000; s++) {
            float du = curand_uniform(&local_rand_state);
            float dv = curand_uniform(&local_rand_state);

            float u = float(i + du) / float(width);
            float v = float(j + dv) / float(height);

            Ray ray = camera->generate_ray(u, v);
            color += trace(ray, camera, scene, &local_rand_state);
        }

        color /= 2000.0f;
        color = vec3::clamp(color, 0.0f, 1.0f);
        color = vec3(sqrt(color.r), sqrt(color.g), sqrt(color.b));
        framebuffer[pixel_index] = color;
    }

    GPU_FUNC vec3 trace(const Ray& ray, Camera* camera, Scene* scene,
                        curandState* rand_state) {
        Ray current_ray = ray;
        vec3 current_attenuation = vec3(1.0f);

        for (int i = 0; i < 10; i++) {
            HitRecord record;
            if (scene->intersect(current_ray, 0.001f, 10000.0f, record)) {
                Ray scattered;
                vec3 attenuation;
                if (record.material->scatter(current_ray, record, attenuation, scattered,
                                             rand_state)) {
                    current_ray = scattered;
                    current_attenuation *= attenuation;
                } else {
                    return current_attenuation * attenuation;
                }
            } else {
                vec3 unit_direction = vec3::normalized(current_ray.direction);
                float t = 0.5f * (unit_direction.y + 1.0f);
                vec3 background_color = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) +
                                        t * vec3(0.5f, 0.7f, 1.0f);
                return current_attenuation * background_color;
            }
        }

        return current_attenuation;
    }
};

} // namespace rtx
