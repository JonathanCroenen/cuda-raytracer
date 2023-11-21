#pragma once

#include "materials/material.h"
#include "math/vec3.h"
#include "primitives/volume.h"
#include "scene.h"
#include <memory>

namespace rtx::scene {

std::unique_ptr<Scene> create_cornell_triangles() {
    using vec3 = math::vec3<float>;

    auto scene = std::make_unique<Scene>();
    MaterialId white = scene->register_material(Lambertian(vec3(0.73f, 0.73f, 0.73f)));
    MaterialId green = scene->register_material(Lambertian(vec3(0.12f, 0.45f, 0.15f)));
    MaterialId blue = scene->register_material(Lambertian(vec3(0.12f, 0.15f, 0.45f)));
    MaterialId red = scene->register_material(Lambertian(vec3(0.65f, 0.05f, 0.05f)));
    MaterialId light = scene->register_material(Emissive(vec3(15.0f)));

    scene->add_volume(Plane(vec3(0.0f, -5.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f)), white)
        .add_volume(Plane(vec3(5.0f, 0.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f)), red)
        .add_volume(Plane(vec3(-5.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)), green)
        .add_volume(Plane(vec3(0.0f, 0.0f, -5.0f), vec3(0.0f, 0.0f, 1.0f)), blue)
        .add_volume(Plane(vec3(0.0f, 5.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)), white)
        .add_volume(Triangle(vec3(0.0f, 4.95f, -4.0f), vec3(-1.5f, 4.95f, -2.8f),
                             vec3(1.5f, 4.95f, -2.8f)),
                    light)
        .add_volume(Triangle(vec3(0.0f, 1.0f, -3.0f),
                             vec3(1.0f, -0.0f, -1.2f), vec3(2.7f, 1.8f, -3.0f)),
                    white)
        .add_volume(Triangle(vec3(0.0f, -2.0f, -4.0f), vec3(-2.7f, 0.4f, -4.3f),
                             vec3(-1.6f, -2.2f, -3.0f)), red);

    scene->build();

    return scene;
}

} // namespace rtx::scene
