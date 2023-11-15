#include "primitives/volume.h"

namespace rtx {

Volume::~Volume() {
    switch (type) {
    case Type::SPHERE:
        _data.sphere.~Sphere();
        break;
    case Type::PLANE:
        _data.plane.~Plane();
        break;
    }
}


GPU_FUNC bool Volume::intersect(const Ray& ray, float t_min, float t_max,
                                HitRecord& record) const {
    switch (type) {
    case Type::SPHERE:
        return _data.sphere.intersect(ray, t_min, t_max, record);
    case Type::PLANE:
        return _data.plane.intersect(ray, t_min, t_max, record);
    }

    return false;
}

} // namespace rtx
