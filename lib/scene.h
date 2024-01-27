#pragma once

#include "math/vec3.h"
#include "primitives/volume.h"
#include "materials/material.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"
#include <vector>
#include <memory>

namespace rtx {

typedef unsigned int MaterialId;

class Scene : public utils::GPUManaged {
public:
    static std::unique_ptr<Scene> create();
    ~Scene();

    Scene& add_volume(const Volume& volume, MaterialId material);
    MaterialId register_material(const Material& material);
    void build();

    GPU_FUNC inline const Material* get_material(int id) const { return &_cuda_materials[id]; }
    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                              HitRecord& record) const;

private:
    Scene() = default;

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
