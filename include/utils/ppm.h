#pragma once

#include "math/vec3.h"

namespace rtx {

template <typename T>
inline void clamp_ppm(math::vec3<T>* framebuffer, int width, int height) {
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            int pixel_index = j * width + i;

            framebuffer[pixel_index] = math::vec3<T>::clamp(framebuffer[pixel_index],
                                                            0.0f, 1.0f);
        }
    }
}

template <typename T>
inline void print_ppm(math::vec3<T>* framebuffer, int width, int height) {
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {

            int pixel_index = j * width + i;
            math::vec3 pixel = framebuffer[pixel_index];

            int r = int(255.99f * pixel.r);
            int g = int(255.99f * pixel.g);
            int b = int(255.99f * pixel.b);

            std::cout << r << " " << g << " " << b << "\n";
        }
    }
}

} // namespace rtx
