#pragma once

#include "math/vec3.h"
#include "utils/cuda.h"

namespace rtx {

KERNEL_FUNC void gaussian_filter(
    math::vec3<float>* buffer_in, math::vec3<float>* buffer_out, int width, int height) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= width || y >= height)
    return;

  int pixel_index = y * width + x;
  math::vec3<float> pixel = buffer_in[pixel_index];

  math::vec3<float> color(0.0f);
  float filter[3][3] = {{1.0f, 2.0f, 1.0f}, {2.0f, 4.0f, 2.0f}, {1.0f, 2.0f, 1.0f}};

  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      int x_ = x + i;
      int y_ = y + j;

      if (x_ < 0 || x_ >= width || y_ < 0 || y_ >= height)
        continue;

      int pixel_index_ = y_ * width + x_;
      math::vec3<float> pixel_ = buffer_in[pixel_index_];

      float v = filter[i + 1][j + 1];

      color += v * pixel_;
    }
  }

  buffer_out[pixel_index] = color / 16.0f;
}

} // namespace rtx
