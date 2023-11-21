#pragma once

#include "math/vec3.h"
#include "primitives/volume.h"
#include "materials/material.h"
#include "light.h"
#include "camera.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"
#include <vector>

namespace rtx {

typedef unsigned int MaterialId;

class Scene : public utils::GPUManaged {
public:
    Scene() {}
    ~Scene();

    Scene& add_volume(const Volume& volume, MaterialId material);
    MaterialId register_material(const Material& material);
    GPU_FUNC inline const Material* get_material(int id) const { return &_cuda_materials[id]; }

    void build();

    __device__ bool intersect(const Ray& ray, float t_min, float t_max,
                              HitRecord& record) const;

private:
    struct Object {
        Object(const Volume& volume, MaterialId material_id)
            : volume(volume), material_id(material_id) {}
        Volume volume;
        MaterialId material_id;
    };

    std::vector<Object> _objects;
    std::vector<Material> _materials;

    Object* _cuda_objects;
    size_t _num_objects;

    Material* _cuda_materials;
    size_t _num_materials;
};

} // namespace rtx
