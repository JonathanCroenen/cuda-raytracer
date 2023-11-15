#pragma once

#include "math/vec3.h"
#include "primitives/sphere.h"
#include "primitives/plane.h"
#include "ray.h"
#include "utils/cuda.h"
#include "utils/gpu_allocated.h"

namespace rtx {

class Volume : public GPUAllocated {
public:
    Volume(const Sphere& sphere) : type(Type::SPHERE) { _data.sphere = sphere; }
    Volume(const Plane& plane) : type(Type::PLANE) { _data.plane = plane; }

    ~Volume();

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    enum class Type { SPHERE, PLANE } type;

    union Data {
        Data() {}
        ~Data() {}

        Sphere sphere;
        Plane plane;
    } _data;
};

template <typename T, typename... Args> Volume make_volume(Args... args) {
    return Volume(T(args...));
}

} // namespace rtx
