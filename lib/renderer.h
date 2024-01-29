#pragma once

#include "camera.h"
#include "math/vec3.h"
#include "scene.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"
#include <curand_kernel.h>
#include <memory>

namespace rtx {

class Renderer : public utils::GPUManaged {
private:
  using vec3 = math::vec3<float>;

public:
  ~Renderer();

  struct Parameters {
    uint width;
    uint height;
    uint rays_per_pixel;
    uint max_depth;
    dim3 num_threads;
  };

  static std::unique_ptr<Renderer> create(const Parameters& params);

  vec3* render(std::unique_ptr<Camera> camera, std::unique_ptr<Scene> scene);

private:
  Renderer(Parameters params);

  friend KERNEL_FUNC void init_random_state_kernel(Renderer* renderer);
  friend KERNEL_FUNC void render_kernel(Renderer* renderer, Camera* camera, Scene* scene);

  GPU_FUNC void render(Camera* camera, Scene* scene, uint i, uint j);
  GPU_FUNC vec3 trace(const Ray& ray, Camera* camera, Scene* scene, curandState* rand_state);

private:
  uint _width;
  uint _height;
  uint _rays_per_pixel;
  uint _max_depth;
  dim3 _num_threads;
  dim3 _num_blocks;

  curandState* _random_state;
  vec3* _framebuffer;
};

} // namespace rtx
