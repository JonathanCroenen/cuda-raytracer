#include "camera.h"
#include "light.h"
#include "math/vec3.h"
#include "primitives/sphere.h"
#include "primitives/volume.h"
#include "materials/material.h"
#include "ray.h"
#include "scene.h"
#include "utils/cuda.h"
#include "utils/ppm.h"
#include <curand_kernel.h>

using namespace rtx;
using vec3f = math::vec3<float>;

GPU_FUNC vec3f trace(const Ray& ray, Camera* camera, Scene* scene, curandState* rand_state) {
    Ray current_ray = ray;
    vec3f current_attenuation = vec3f(1.0f);

    for (int i = 0; i < 50; i++) {
        HitRecord record;
        if (scene->intersect(current_ray, 0.001f, 10000.0f, record)) {
            Ray scattered;
            vec3f attenuation;
            if (record.material->scatter(current_ray, record, attenuation, scattered,
                                         rand_state)) {
                current_ray = scattered;
                current_attenuation *= attenuation;
            } else {
                return current_attenuation * attenuation;
            }
        } else {
            vec3f unit_direction = vec3f::normalized(current_ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            vec3f background_color = vec3f(0.01f);
                // (1.0f - t) * vec3f(1.0f, 1.0f, 1.0f) +
                //                      t * vec3f(0.5f, 0.7f, 1.0f);
            return current_attenuation * background_color;
        }
    }

    return current_attenuation;
}

KERNEL_FUNC void render(vec3f* framebuffer, int width, int height, Camera* camera,
                        Scene* scene, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) {
        return;
    }

    int pixel_index = j * width + i;
    curandState local_rand_state = rand_state[pixel_index];

    vec3f color = vec3f(0.0f);
    for (int s = 0; s < 10000; s++) {
        float du = curand_uniform(&local_rand_state);
        float dv = curand_uniform(&local_rand_state);

        float u = float(i + du) / float(width);
        float v = float(j + dv) / float(height);

        Ray ray = camera->generate_ray(u, v);
        color += trace(ray, camera, scene, &local_rand_state);
    }

    color /= 10000.0f;
    color = vec3f(sqrt(color.r), sqrt(color.g), sqrt(color.b));
    framebuffer[pixel_index] = color;
}

KERNEL_FUNC void render_init(int width, int height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) {
        return;
    }

    int pixel_index = j * width + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

int main() {
    int width = 800;
    int height = 600;

    int tx = 8;
    int ty = 8;

    size_t num_bytes = width * height * sizeof(vec3f);

    vec3f* framebuffer;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&framebuffer, num_bytes));

    curandState* rand_state;
    CHECK_CUDA_ERRORS(cudaMalloc(&rand_state, width * height * sizeof(curandState)));

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(width, height, rand_state);

    Material glass = make_material<Dielectric>(1.5f);
    Material red_metal = make_material<Metal>(vec3f(0.8f, 0.3f, 0.3f), 0.0f);
    Material diffuse = make_material<Lambertian>(vec3f(0.8f, 0.8f, 0.0f));
    Material light = make_material<Emissive>(vec3f(15.0f, 15.0f, 15.0f));

    Scene* scene = new Scene{
        make_volume<Sphere>(vec3f(1.0f, -0.3f, -1.5f), 0.3f, glass),
        make_volume<Sphere>(vec3f(-2.5f, -0.3f, -2.0f), 0.8f, red_metal),
        make_volume<Sphere>(vec3f(2.5f, 1.3f, -2.0f), 0.8f, light),
        make_volume<Plane>(vec3f(0.0f, -10.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f),
                           diffuse),
    };

    Camera* camera = new Camera(vec3f(0.0f, 0.0f, 1.0f), vec3f(0.0f, 0.0f, -1.0f),
                                vec3f(0.0f, 1.0f, 0.0f), 90.0f,
                                float(width) / float(height));

    render<<<blocks, threads>>>(framebuffer, width, height, camera, scene, rand_state);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    clamp_ppm(framebuffer, width, height);
    print_ppm(framebuffer, width, height);

    delete camera;
    delete scene;

    CHECK_CUDA_ERRORS(cudaFree(rand_state));
    CHECK_CUDA_ERRORS(cudaFree(framebuffer));

    return 0;
}
