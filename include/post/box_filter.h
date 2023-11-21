#pragma once

#include "math/vec3.h"
#include "utils/cuda.h"

namespace rtx {

KERNEL_FUNC void box_filter(math::vec3<float>* buffer_in, math::vec3<float>* buffer_out,
                           int width, int height, int radius) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    int pixel_index = y * width + x;

    math::vec3<float> color(0.0f);

    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            int x_ = x + i;
            int y_ = y + j;

            if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height)
                continue;

            int pixel_index_ = y_ * width + x_;
            math::vec3<float> pixel = buffer_in[pixel_index_];

            color += pixel;
        }
    }

    int num_pixels = (2 * radius + 1) * (2 * radius + 1);
    buffer_out[pixel_index] = color / float(num_pixels);
}

} // namespace rtx
