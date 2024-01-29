#include "primitives/volume.h"

namespace rtx {

GPU_FUNC bool Volume::intersect(
    const Ray& ray, float t_min, float t_max, HitRecord& record) const {
  if (is<Sphere>()) {
    return get<Sphere>().intersect(ray, t_min, t_max, record);
  } else if (is<Plane>()) {
    return get<Plane>().intersect(ray, t_min, t_max, record);
  } else if (is<Triangle>()) {
    return get<Triangle>().intersect(ray, t_min, t_max, record);
  } else if (is<Disk>()) {
    return get<Disk>().intersect(ray, t_min, t_max, record);
  } else if (is<Quad>()) {
    return get<Quad>().intersect(ray, t_min, t_max, record);
  } else {
    return false;
  }
}

} // namespace rtx
