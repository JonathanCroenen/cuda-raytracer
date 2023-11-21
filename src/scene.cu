#include "scene.h"
#include <iostream>

namespace rtx {

Scene& Scene::add_volume(const Volume& volume, MaterialId material) {
    _objects.emplace_back(volume, material);
    return *this;
}

MaterialId Scene::register_material(const Material& material) {
    _materials.push_back(material);
    return _materials.size() - 1;
}

void Scene::build() {
    _num_objects = _objects.size();
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&_cuda_objects, sizeof(Object) * _num_objects));
    CHECK_CUDA_ERRORS(cudaMemcpy(_cuda_objects, _objects.data(), sizeof(Object) * _num_objects,
                                 cudaMemcpyHostToDevice));

    _num_materials = _materials.size();
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&_cuda_materials, sizeof(Material) * _num_materials));
    CHECK_CUDA_ERRORS(cudaMemcpy(_cuda_materials, _materials.data(),
                                 sizeof(Material) * _num_materials, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

Scene::~Scene() {
    CHECK_CUDA_ERRORS(cudaFree(_cuda_objects));
    CHECK_CUDA_ERRORS(cudaFree(_cuda_materials));
}

GPU_FUNC bool Scene::intersect(const Ray& ray, float t_min, float t_max,
                               HitRecord& record) const {
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < _num_objects; ++i) {
        if (_cuda_objects[i].volume.intersect(ray, t_min, closest_so_far, record)) {
            hit_anything = true;
            closest_so_far = record.t;
            record.material = get_material(_cuda_objects[i].material_id);
        }
    }

    return hit_anything;
}

} // namespace rtx
