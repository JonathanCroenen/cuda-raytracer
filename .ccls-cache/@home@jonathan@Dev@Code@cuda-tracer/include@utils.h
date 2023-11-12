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
