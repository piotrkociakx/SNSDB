#include "cuda_encryption.h"

#ifdef USE_CUDA

// CUDA kernel for XOR encryption
__global__ void xor_encrypt_kernel(uint8_t* data, const uint8_t* key, size_t data_size, size_t key_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        data[idx] ^= key[idx % key_size];
    }
}

#endif
