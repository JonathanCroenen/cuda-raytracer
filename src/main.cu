#include "camera.h"
#include "light.h"
#include "materials/material.h"
#include "math/vec3.h"
#include "post/box_filter.h"
#include "primitives/volume.h"
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

    for (int i = 0; i < 10; i++) {
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
            vec3f background_color = (1.0f - t) * vec3f(1.0f, 1.0f, 1.0f) +
                                     t * vec3f(0.5f, 0.7f, 1.0f);
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
    for (int s = 0; s < 1000; s++) {
        float du = curand_uniform(&local_rand_state);
        float dv = curand_uniform(&local_rand_state);

        float u = float(i + du) / float(width);
        float v = float(j + dv) / float(height);

        Ray ray = camera->generate_ray(u, v);
        color += trace(ray, camera, scene, &local_rand_state);
    }

    color /= 1000.0f;
    color = vec3f::clamp(color, 0.0f, 1.0f);
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
    vec3f* filtered_framebuffer;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&filtered_framebuffer, num_bytes));

    curandState* rand_state;
    CHECK_CUDA_ERRORS(cudaMalloc(&rand_state, width * height * sizeof(curandState)));

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(width, height, rand_state);

    Scene* scene = new Scene;
    MaterialId white = scene->register_material(
        Lambertian(vec3f(0.73f, 0.73f, 0.73f)));
    MaterialId green = scene->register_material(
        Lambertian(vec3f(0.12f, 0.45f, 0.15f)));
    MaterialId blue = scene->register_material(
        Lambertian(vec3f(0.12f, 0.15f, 0.45f)));
    MaterialId red = scene->register_material(
        Lambertian(vec3f(0.65f, 0.05f, 0.05f)));
    MaterialId light = scene->register_material(Emissive(vec3f(15.0f)));
    MaterialId metal = scene->register_material(Metal(vec3f(1.0f), 0.0f));
    MaterialId glass = scene->register_material(Dielectric(1.5f));

    // create the materials and scene for a simple cornell box that contains 1 emissive
    // sphere and 1 metal sphere

    scene
        ->add_volume(
            Plane(vec3f(0.0f, -5.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f)), white)
        .add_volume(Plane(vec3f(5.0f, 0.0f, 0.0f), vec3f(-1.0f, 0.0f, 0.0f)),
                    red)
        .add_volume(Plane(vec3f(-5.0f, 0.0f, 0.0f), vec3f(1.0f, 0.0f, 0.0f)),
                    green)
        .add_volume(Plane(vec3f(0.0f, 0.0f, -5.0f), vec3f(0.0f, 0.0f, 1.0f)),
                    blue)
        .add_volume(Plane(vec3f(0.0f, 5.0f, 0.0f), vec3f(0.0f, -1.0f, 0.0f)),
                    white)
        .add_volume(Sphere(vec3f(0.0f, 5.60f, -2.5f), 1.00), light)
        .add_volume(Sphere(vec3f(-1.0f, 1.0f, -2.5f), 1.00), white)
        .add_volume(Sphere(vec3f(1.0f, -0.5f, -3.5f), 1.00), metal)
        .add_volume(Sphere(vec3f(-1.4f, -1.0f, -1.5f), 1.00), glass);

    scene->build();

    Camera* camera = new Camera(vec3f(0.0f, 0.0f, 3.0f), vec3f(0.0f, 0.0f, -1.0f),
                                vec3f(0.0f, 1.0f, 0.0f), 90.0f,
                                float(width) / float(height));

    render<<<blocks, threads>>>(framebuffer, width, height, camera, scene, rand_state);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    // box_filter<<<blocks, threads>>>(framebuffer, filtered_framebuffer, width, height, 3);
    // CHECK_CUDA_ERRORS(cudaGetLastError());
    // CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    // TODO: weird bug where corrupted values sometimes appear in the framebuffer
    print_ppm(framebuffer, width, height);

    delete camera;
    delete scene;

    CHECK_CUDA_ERRORS(cudaFree(rand_state));
    CHECK_CUDA_ERRORS(cudaFree(framebuffer));
    CHECK_CUDA_ERRORS(cudaFree(filtered_framebuffer));

    return 0;
}
