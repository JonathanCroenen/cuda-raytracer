#include "primitives/volume.h"

namespace rtx {

GPU_FUNC bool Volume::intersect(
    const Ray& ray, float t_min, float t_max, HitRecord& record) const {
  switch (type_id()) {
  case type_id_of<Sphere>():
    return get<Sphere>().intersect(ray, t_min, t_max, record);
  case type_id_of<Plane>():
    return get<Plane>().intersect(ray, t_min, t_max, record);
  case type_id_of<Triangle>():
    return get<Triangle>().intersect(ray, t_min, t_max, record);
  case type_id_of<Disk>():
    return get<Disk>().intersect(ray, t_min, t_max, record);
  case type_id_of<Quad>():
    return get<Quad>().intersect(ray, t_min, t_max, record);
  default:
    return false;
  }
}

} // namespace rtx
