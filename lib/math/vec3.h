#pragma once

#include "utils/cuda.h"
#include <curand_kernel.h>

namespace rtx::math {

template <typename T = float>
struct vec3 {
  CPU_GPU_FUNC vec3() : x(0), y(0), z(0) {}
  CPU_GPU_FUNC vec3(T e0) : x(e0), y(e0), z(e0) {}
  CPU_GPU_FUNC vec3(T e0, T e1, T e2) : x(e0), y(e1), z(e2) {}

  CPU_GPU_FUNC inline const vec3<T>& operator+() const { return *this; }
  CPU_GPU_FUNC inline vec3<T> operator-() const { return vec3<T>(-x, -y, -z); }
  CPU_GPU_FUNC inline T operator[](int i) const { return (&x)[i]; }
  CPU_GPU_FUNC inline T& operator[](int i) { return (&x)[i]; }

  CPU_GPU_FUNC inline vec3<T>& operator+=(const vec3<T>& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  CPU_GPU_FUNC inline vec3<T>& operator-=(const vec3<T>& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  CPU_GPU_FUNC inline vec3<T>& operator*=(const vec3<T>& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
  }
  CPU_GPU_FUNC inline vec3<T>& operator/=(const vec3<T>& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  CPU_GPU_FUNC inline vec3<T>& operator*=(const T t) {
    x *= t;
    y *= t;
    z *= t;
    return *this;
  }
  CPU_GPU_FUNC inline vec3<T>& operator/=(const T t) {
    x /= t;
    y /= t;
    z /= t;
    return *this;
  }

  CPU_GPU_FUNC inline friend vec3<T> operator+(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3<T>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
  }
  CPU_GPU_FUNC inline friend vec3<T> operator-(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3<T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
  }
  CPU_GPU_FUNC inline friend vec3<T> operator*(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3<T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
  }
  CPU_GPU_FUNC inline friend vec3<T> operator/(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3<T>(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
  }
  CPU_GPU_FUNC inline friend vec3<T> operator*(const vec3<T>& v, const T t) {
    return vec3<T>(v.x * t, v.y * t, v.z * t);
  }
  CPU_GPU_FUNC inline friend vec3<T> operator*(const T t, const vec3<T>& v) {
    return vec3<T>(v.x * t, v.y * t, v.z * t);
  }
  CPU_GPU_FUNC inline friend vec3<T> operator/(const vec3<T>& v, const T t) {
    return vec3<T>(v.x / t, v.y / t, v.z / t);
  }

  CPU_GPU_FUNC inline static T dot(const vec3<T>& v1, const vec3<T>& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  }
  CPU_GPU_FUNC inline static vec3<T> cross(const vec3<T>& v1, const vec3<T>& v2) {
    return vec3<T>(
        v1.y * v2.z - v1.z * v2.y, -(v1.x * v2.z - v1.z * v2.x), v1.x * v2.y - v1.y * v2.x);
  }
  CPU_GPU_FUNC inline static vec3<T> normalized(const vec3<T>& v) {
    return v / v.length();
  }
  CPU_GPU_FUNC inline static vec3<T> clamp(const vec3<T>& v, T min, T max) {
    return vec3<T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
  }
  CPU_GPU_FUNC inline static vec3<T> reflect(const vec3<T>& v, const vec3<T>& n) {
    vec3 normal = normalized(n);
    return v - 2 * dot(v, normal) * normal;
  }
  CPU_GPU_FUNC inline static T angle(const vec3<T>& v1, const vec3<T>& v2) {
    return acos(dot(v1, v2) / (v1.length() * v2.length()));
  }

  GPU_FUNC inline static vec3<T> random(curandState* rand_state) {
    return vec3<T>(
        curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state));
  }
  GPU_FUNC inline static vec3<T> random(T min, T max, curandState* rand_state) {
    return vec3<T>::random(rand_state) * (max - min) + min;
  }
  GPU_FUNC inline static vec3<T> random_unit(curandState* rand_state) {
    T a = curand_uniform(rand_state) * 2 * M_PI;
    T z = curand_uniform(rand_state) * 2 - 1;
    T r = sqrt(1 - z * z);
    return vec3<T>(r * cos(a), r * sin(a), z);
  }
  inline static vec3<T> random() {
    return vec3<T>(
        rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0));
  }
  inline static vec3<T> random(T min, T max) {
    return vec3<T>::random() * (max - min) + min;
  }
  inline static vec3<T> random_unit() {
    T a = rand() / (RAND_MAX + 1.0) * 2 * M_PI;
    T z = rand() / (RAND_MAX + 1.0) * 2 - 1;
    T r = sqrt(1 - z * z);
    return vec3<T>(r * cos(a), r * sin(a), z);
  }

  CPU_GPU_FUNC inline T length() const { return sqrt(x * x + y * y + z * z); }
  CPU_GPU_FUNC inline T squared_length() const { return x * x + y * y + z * z; }

  union {
    struct {
      T x, y, z;
    };
    struct {
      T r, g, b;
    };
  };

private:
  CPU_GPU_FUNC inline static T clamp(T v, T min, T max) {
    if (v < min)
      return min;
    if (v > max)
      return max;
    return v;
  }
};

} // namespace rtx::math
