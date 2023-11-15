#include "scene.h"

#include <iostream>

namespace rtx {

Scene::Scene(std::initializer_list<Volume> volumes) : _num_volumes(volumes.size()) {
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&_volumes, sizeof(Volume) * _num_volumes));
    CHECK_CUDA_ERRORS(cudaMemcpy(_volumes, volumes.begin(),
                                 sizeof(Volume) * _num_volumes, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

Scene::~Scene() {
    CHECK_CUDA_ERRORS(cudaFree(_volumes));
}

GPU_FUNC bool Scene::intersect(const Ray& ray, float t_min, float t_max,
                                 HitRecord& record) const {
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < _num_volumes; ++i) {
        if (_volumes[i].intersect(ray, t_min, closest_so_far, record)) {
            hit_anything = true;
            closest_so_far = record.t;
        }
    }

    return hit_anything;
}

} // namespace rtx
