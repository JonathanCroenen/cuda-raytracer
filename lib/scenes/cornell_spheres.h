#pragma once

#include "materials/material.h"
#include "math/vec3.h"
#include "primitives/volume.h"
#include "scene.h"
#include <memory>

namespace rtx::scene {

std::unique_ptr<Scene> create_cornell_spheres() {
  using vec3 = math::vec3<float>;

  auto scene = Scene::create();
  MaterialId white = scene->register_material(Lambertian(vec3(0.73f, 0.73f, 0.73f)));
  MaterialId green = scene->register_material(Lambertian(vec3(0.12f, 0.45f, 0.15f)));
  MaterialId blue = scene->register_material(Lambertian(vec3(0.12f, 0.15f, 0.45f)));
  MaterialId red = scene->register_material(Lambertian(vec3(0.65f, 0.05f, 0.05f)));
  MaterialId light = scene->register_material(Emissive(vec3(15.0f)));
  MaterialId metal = scene->register_material(Metal(vec3(1.0f), 0.0f));
  MaterialId glass = scene->register_material(Dielectric(1.5f));

  scene->add_volume(Plane(vec3(0.0f, -5.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f)), white)
      .add_volume(Plane(vec3(5.0f, 0.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f)), red)
      .add_volume(Plane(vec3(-5.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)), green)
      .add_volume(Plane(vec3(0.0f, 0.0f, -5.0f), vec3(0.0f, 0.0f, 1.0f)), blue)
      .add_volume(Plane(vec3(0.0f, 5.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)), white)
      .add_volume(Sphere(vec3(0.0f, 5.60f, -2.5f), 1.00), light)
      .add_volume(Sphere(vec3(-1.0f, 1.0f, -2.5f), 1.00), white)
      .add_volume(Sphere(vec3(1.0f, -0.5f, -1.5f), 1.00), metal)
      .add_volume(Sphere(vec3(-1.4f, -1.0f, -1.5f), 1.00), glass);

  scene->build();

  return scene;
}

} // namespace rtx::scene
