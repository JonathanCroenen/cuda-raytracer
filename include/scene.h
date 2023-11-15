#pragma once

#include "math/vec3.h"
#include "primitives/volume.h"
#include "light.h"
#include "camera.h"
#include "utils/cuda.h"
#include "utils/gpu_allocated.h"
#include <initializer_list>

namespace rtx {

class Scene : public GPUAllocated {
public:
    Scene(std::initializer_list<Volume> volumes);
    ~Scene();

    __device__ bool intersect(const Ray& ray, float t_min, float t_max,
                              HitRecord& record) const;

private:
    Volume* _volumes;
    size_t _num_volumes;
};

} // namespace rtx
