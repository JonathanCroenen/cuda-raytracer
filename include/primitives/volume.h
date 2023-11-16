#pragma once

#include "primitives/plane.h"
#include "primitives/sphere.h"
#include "ray.h"
#include "utils/cuda.h"
#include "utils/gpu_managed.h"

namespace rtx {

class Volume : public utils::GPUManaged {
public:
    Volume(const Sphere& sphere) : _type(Type::SPHERE) { _data.sphere = sphere; }
    Volume(const Plane& plane) : _type(Type::PLANE) { _data.plane = plane; }

    ~Volume();

    GPU_FUNC bool intersect(const Ray& ray, float t_min, float t_max,
                            HitRecord& record) const;

private:
    enum class Type { SPHERE, PLANE } _type;

    union Data {
        Data() {}

        ~Data() {}

        Sphere sphere;
        Plane plane;
    } _data;
};

template <typename T, typename... Args>
Volume make_volume(Args... args) {
    return Volume(T(args...));
}

} // namespace rtx
