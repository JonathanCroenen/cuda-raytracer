#pragma once

#include <iostream>

#define CHECK_CUDA_ERRORS(val) CheckCudaErrors((val), #val, __FILE__, __LINE__)

void CheckCudaErrors(cudaError_t result,
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


class GPUAllocated {
public:
    void* operator new(size_t size) {
        void* ptr;
        CHECK_CUDA_ERRORS(cudaMallocManaged(&ptr, size));
        return ptr;
    }

    void* operator new[](size_t size) {
        void* ptr;
        CHECK_CUDA_ERRORS(cudaMallocManaged(&ptr, size));
        return ptr;
    }

    void operator delete(void* ptr) {
        CHECK_CUDA_ERRORS(cudaFree(ptr));
    }

    void operator delete[](void* ptr) {
        CHECK_CUDA_ERRORS(cudaFree(ptr));
    }

protected:
    GPUAllocated() = default;
};
