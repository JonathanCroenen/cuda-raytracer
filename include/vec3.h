#pragma once

namespace rtx {

template <typename T = float> struct vec3 {
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(T e0, T e1, T e2) : x(e0), y(e1), z(e2) {}

    __host__ __device__ inline const vec3<T>& operator+() const { return *this; }
    __host__ __device__ inline vec3<T> operator-() const { return vec3<T>(-x, -y, -z); }
    __host__ __device__ inline T operator[](int i) const { return (&x)[i]; }
    __host__ __device__ inline T& operator[](int i) { return (&x)[i]; }

    __host__ __device__ inline vec3<T>& operator+=(const vec3<T>& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __host__ __device__ inline vec3<T>& operator-=(const vec3<T>& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    __host__ __device__ inline vec3<T>& operator*=(const vec3<T>& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    __host__ __device__ inline vec3<T>& operator/=(const vec3<T>& v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    __host__ __device__ inline vec3<T>& operator*=(const T t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }
    __host__ __device__ inline vec3<T>& operator/=(const T t) {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }

    __host__ __device__ inline friend vec3<T> operator+(const vec3<T>& v1,
                                                        const vec3<T>& v2) {
        return vec3<T>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }
    __host__ __device__ inline friend vec3<T> operator-(const vec3<T>& v1,
                                                        const vec3<T>& v2) {
        return vec3<T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }
    __host__ __device__ inline friend vec3<T> operator*(const vec3<T>& v1,
                                                        const vec3<T>& v2) {
        return vec3<T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    }
    __host__ __device__ inline friend vec3<T> operator/(const vec3<T>& v1,
                                                        const vec3<T>& v2) {
        return vec3<T>(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
    }
    __host__ __device__ inline friend vec3<T> operator*(const vec3<T>& v, const T t) {
        return vec3<T>(v.x * t, v.y * t, v.z * t);
    }
    __host__ __device__ inline friend vec3<T> operator*(const T t, const vec3<T>& v) {
        return vec3<T>(v.x * t, v.y * t, v.z * t);
    }
    __host__ __device__ inline friend vec3<T> operator/(const vec3<T>& v, const T t) {
        return vec3<T>(v.x / t, v.y / t, v.z / t);
    }

    __host__ __device__ inline static T dot(const vec3<T>& v1, const vec3<T>& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
    __host__ __device__ inline static vec3<T> cross(const vec3<T>& v1,
                                                    const vec3<T>& v2) {
        return vec3<T>(v1.y * v2.z - v1.z * v2.y,
                       -(v1.x * v2.z - v1.z * v2.x),
                       v1.x * v2.y - v1.y * v2.x);
    }
    __host__ __device__ inline static vec3<T> normalized(const vec3<T>& v) {
        return v / v.length();
    }

    __host__ __device__ inline T length() const { return sqrt(x * x + y * y + z * z); }
    __host__ __device__ inline T squared_length() const { return x * x + y * y + z * z; }
    __host__ __device__ inline vec3<T>& normalize() {
        T k = 1.0 / sqrt(x * x + y * y + z * z);
        x *= k;
        y *= k;
        z *= k;
        return *this;
    }

    union {
        struct {
            T x, y, z;
        };
        struct {
            T r, g, b;
        };
    };
};

} // namespace rtx

