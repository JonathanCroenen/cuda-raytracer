#include <iostream>

#include "camera.h"
#include "utils.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "light.h"
#include "scene.h"

using namespace rtx;
using vec3f = rtx::vec3<float>;

__device__ vec3f trace(const Ray& r, Camera* camera, Scene* scene) {
    HitRecord rec;
    if (scene->hit(r, 0.0f, 1000.0f, rec)) {
        return vec3f::clamp(vec3f(0.2f) + scene->calc_light(rec, *camera), 0.0f, 1.0f);
    }

    // float t = 0.5f * (r.direction.y + 1.0f);
    // return (1.0f - t) * vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return vec3f(0.05, 0.1f, 0.15f);
}

__global__ void
render(vec3f* framebuffer, int width, int height, Camera* camera, Scene* scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) {
        return;
    }

    int pixel_index = j * width + i;
    float u = float(i) / float(width);
    float v = float(j) / float(height);

    Ray ray = camera->generate_ray(u, v);
    framebuffer[pixel_index] = trace(ray, camera, scene);
}

void print_ppm(vec3f* framebuffer, int width, int height) {
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

    size_t num_bytes = width * height * sizeof(vec3f);

    vec3f* framebuffer;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&framebuffer, num_bytes));

    int tx = 8;
    int ty = 8;

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    Scene* scene = new Scene();
    scene->add(Sphere(vec3f(0.0f, 0.0f, -2.0f), 0.5f, vec3f(1.0f, 0.0f, 0.0f)));
    scene->add(Sphere(vec3f(1.0f, -0.3f, -1.5f), 0.3f, vec3f(0.0f, 1.0f, 0.0f)));
    scene->add(Sphere(vec3f(-2.5f, -0.3f, -2.5f), 0.8f, vec3f(0.0f, 0.0f, 1.0f)));
    scene->add(Sphere(vec3f(2.5f, 1.3f, -2.0f), 0.8f, vec3f(0.0f, 1.0f, 1.0f)));
    scene->add(Light(vec3f(3.0f, 1.0f, -0.4f), vec3f(1.0f, 0.3f, 0.0f), 3.0f));
    scene->add(Light(vec3f(-3.0f, -1.0f, -1.0f), vec3f(1.0f, 0.0f, 1.0f), 2.0f));

    Camera* camera = new Camera(vec3f(0.0f, 0.0f, 1.0f),
                                vec3f(0.0f, 0.0f, -1.0f),
                                vec3f(0.0f, 1.0f, 0.0f),
                                90.0f,
                                float(width) / float(height));

    render<<<blocks, threads>>>(framebuffer, width, height, camera, scene);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    print_ppm(framebuffer, width, height);

    delete camera;
    delete scene;

    CHECK_CUDA_ERRORS(cudaFree(framebuffer));

    return 0;
}
