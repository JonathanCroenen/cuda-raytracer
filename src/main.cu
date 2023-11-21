#include "camera.h"
#include "math/vec3.h"
#include "ray.h"
#include "scene.h"
#include "scenes/cornell_quads.h"
#include "utils/cuda.h"
#include "utils/ppm.h"
#include <curand_kernel.h>
#include <memory>

using namespace rtx;
using vec3 = math::vec3<float>;

GPU_FUNC vec3 trace(const Ray& ray, Camera* camera, Scene* scene, curandState* rand_state) {
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

KERNEL_FUNC void render(vec3* framebuffer, int width, int height, Camera* camera,
                        Scene* scene, curandState* rand_state) {
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

    int tx = 16;
    int ty = 16;

    size_t num_bytes = width * height * sizeof(vec3);

    vec3* framebuffer;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&framebuffer, num_bytes));
    vec3* filtered_framebuffer;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&filtered_framebuffer, num_bytes));

    curandState* rand_state;
    CHECK_CUDA_ERRORS(cudaMalloc(&rand_state, width * height * sizeof(curandState)));

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(width, height, rand_state);

    auto scene = scene::create_cornell_quads();

    auto camera = std::make_unique<Camera>(vec3(0.0f, 0.0f, 3.0f), vec3(0.0f, 0.0f, -1.0f),
                                           vec3(0.0f, 1.0f, 0.0f), 90.0f,
                                           float(width) / float(height));

    render<<<blocks, threads>>>(framebuffer, width, height, camera.get(), scene.get(),
                                rand_state);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    // box_filter<<<blocks, threads>>>(framebuffer, filtered_framebuffer, width, height,
    // 3); CHECK_CUDA_ERRORS(cudaGetLastError());
    // CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    // TODO: weird bug where corrupted values sometimes appear in the framebuffer
    print_ppm(framebuffer, width, height);

    CHECK_CUDA_ERRORS(cudaFree(rand_state));
    CHECK_CUDA_ERRORS(cudaFree(framebuffer));
    CHECK_CUDA_ERRORS(cudaFree(filtered_framebuffer));

    return 0;
}
