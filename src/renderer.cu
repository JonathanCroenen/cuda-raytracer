#include "renderer.h"

namespace rtx {

// KERNELS

KERNEL_FUNC void init_random_state_kernel(Renderer* renderer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= renderer->_width || y >= renderer->_height) {
        return;
    }

    int pixel_index = y * renderer->_width + x;
    curand_init(1984 + pixel_index, 0, 0, &renderer->_random_state[pixel_index]);
}

KERNEL_FUNC void render_kernel(Renderer* renderer, Camera* camera, Scene* scene) {
    using vec3 = math::vec3<float>;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= renderer->_width || j >= renderer->_height) {
        return;
    }

    renderer->render(camera, scene, i, j);
}

// RENDERER IMPLEMENTATION

std::unique_ptr<Renderer> Renderer::create(const Parameters& params) {
    // std::make_unique doesn't work here because we use a private constructor
    return std::unique_ptr<Renderer>(new Renderer(params));
}

Renderer::Renderer(Parameters params)
    : _width(params.width),
      _height(params.height),
      _rays_per_pixel(params.rays_per_pixel),
      _max_depth(params.max_depth),
      _num_threads(params.num_threads) {

    CHECK_CUDA_ERRORS(cudaMallocManaged(&_framebuffer, _width * _height * sizeof(vec3)));
    CHECK_CUDA_ERRORS(
        cudaMallocManaged(&_random_state, _width * _height * sizeof(curandState)));

    _num_blocks = dim3(_width / _num_threads.x + 1, _height / _num_threads.y + 1);

    init_random_state_kernel<<<_num_blocks, _num_threads>>>(this);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

Renderer::~Renderer() {
    CHECK_CUDA_ERRORS(cudaFree(_framebuffer));
    CHECK_CUDA_ERRORS(cudaFree(_random_state));
}

math::vec3<float>* Renderer::render(Camera* camera, Scene* scene) {
    render_kernel<<<_num_blocks, _num_threads>>>(this, camera, scene);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    return _framebuffer;
}

GPU_FUNC void Renderer::render(Camera* camera, Scene* scene, uint i, uint j) {
    int pixel_index = j * _width + i;
    curandState local_rand_state = _random_state[pixel_index];

    math::vec3<float> color = math::vec3<float>(0.0f);
    for (int s = 0; s < _rays_per_pixel; s++) {
        float du = curand_uniform(&local_rand_state);
        float dv = curand_uniform(&local_rand_state);

        float u = float(i + du) / float(_width);
        float v = float(j + dv) / float(_height);

        Ray ray = camera->generate_ray(u, v);
        color += trace(ray, camera, scene, &local_rand_state);
    }

    color /= static_cast<float>(_rays_per_pixel);
    color = vec3::clamp(color, 0.0f, 1.0f);
    color = vec3(sqrt(color.r), sqrt(color.g), sqrt(color.b));

    _framebuffer[pixel_index] = color;
}

GPU_FUNC math::vec3<float> Renderer::trace(const Ray& ray, Camera* camera, Scene* scene,
                                           curandState* rand_state) {
    Ray current_ray = ray;
    vec3 current_attenuation = vec3(1.0f);

    for (int i = 0; i < _max_depth; i++) {
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

} // namespace rtx
