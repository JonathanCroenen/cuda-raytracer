#include <iostream>

#include "utils.h"
#include "vec3.h"
#include "ray.h"
#include "scene.h"

using vec3 = rtx::vec3<float>;
using ray = rtx::Ray;

__device__ vec3 trace(const ray& r, rtx::Scene const* const scene) {
    rtx::HitRecord rec;
    if (scene->hit(r, 0.0f, 1000.0f, rec)) {
        return 0.5f * (rec.normal + vec3(1.0f, 1.0f, 1.0f));
    }

    float t = 0.5f * (r.direction.y + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void
render(vec3* framebuffer, int width, int height, rtx::Scene const* const scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) {
        return;
    }

    int pixel_index = j * width + i;
    float u = float(i) / float(width);
    float v = float(j) / float(height);

    ray r(vec3(0.0f, 0.0f, 0.0f),
          vec3(1.33f * (-1.0f + 2.0f * u), -1.0f + 2.0f * v, -1.0f));
    framebuffer[pixel_index] = trace(r, scene);
}

void print_ppm(vec3* framebuffer, int width, int height) {
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {

            int pixel_index = j * width + i;
            vec3 pixel = framebuffer[pixel_index];

            int r = int(255.99f * pixel.r);
            int g = int(255.99f * pixel.g);
            int b = int(255.99f * pixel.b);

            std::cout << r << " " << g << " " << b << "\n";
        }
    }
}

int main() {
    int width = 800;
    int height = 600;

    size_t num_bytes = width * height * sizeof(vec3);

    vec3* framebuffer;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&framebuffer, num_bytes));

    int tx = 8;
    int ty = 8;

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    rtx::Scene* scene;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&scene, sizeof(rtx::Scene)));
    scene->add(rtx::Sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f));
    scene->add(rtx::Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f));

    render<<<blocks, threads>>>(framebuffer, width, height, scene);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    print_ppm(framebuffer, width, height);

    CHECK_CUDA_ERRORS(cudaFree(scene));
    CHECK_CUDA_ERRORS(cudaFree(framebuffer));

    return 0;
}
