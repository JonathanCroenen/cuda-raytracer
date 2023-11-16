#pragma once

#include "utils/cuda.h"

namespace rtx::utils {

class GPUManaged {
public:
    GPUManaged(const GPUManaged&) = delete;

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
    GPUManaged() = default;
};

} // namespace rtx::utils
