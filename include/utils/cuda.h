#pragma once

#include <iostream>


#define GPU_FUNC __device__
#define CPU_FUNC __host__
#define CPU_GPU_FUNC __device__ __host__
#define KERNEL_FUNC __global__


#define CHECK_CUDA_ERRORS(val) CheckCudaErrors((val), #val, __FILE__, __LINE__)

inline void CheckCudaErrors(cudaError_t result,
                     const char* const func,
                     const char* const file,
                     const int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(result) << " in \"" << func
                  << "\" (" << file << ":" << line << ")" << std::endl;

        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

