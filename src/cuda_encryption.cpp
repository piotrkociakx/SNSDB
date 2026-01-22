#include "cuda_encryption.h"
#include <algorithm>
#include <iostream>

#ifdef USE_CUDA
// CUDA kernel for XOR encryption
__global__ void xor_encrypt_kernel(uint8_t* data, const uint8_t* key, size_t data_size, size_t key_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        data[idx] ^= key[idx % key_size];
    }
}
#endif

namespace snsdb {

CudaEncryption::CudaEncryption() : cuda_available_(false) {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaSuccess && device_count > 0) {
        cuda_available_ = true;
        std::cout << "CUDA available with " << device_count << " device(s)" << std::endl;
    } else {
        std::cout << "CUDA not available" << std::endl;
    }
#else
    std::cout << "CUDA support not compiled (USE_CUDA not defined)" << std::endl;
#endif
}

CudaEncryption::~CudaEncryption() {
}

std::vector<uint8_t> CudaEncryption::encrypt(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key) {
    std::vector<uint8_t> result = data;
    
#ifdef USE_CUDA
    if (cuda_available_ && !data.empty() && !key.empty()) {
        uint8_t* d_data;
        uint8_t* d_key;
        
        // Allocate device memory
        cudaMalloc(&d_data, data.size());
        cudaMalloc(&d_key, key.size());
        
        // Copy data to device
        cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (data.size() + block_size - 1) / block_size;
        xor_encrypt_kernel<<<grid_size, block_size>>>(d_data, d_key, data.size(), key.size());
        
        // Copy result back
        cudaMemcpy(result.data(), d_data, data.size(), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_data);
        cudaFree(d_key);
    } else
#endif
    {
        // CPU fallback
        for (size_t i = 0; i < result.size(); i++) {
            result[i] ^= key[i % key.size()];
        }
    }
    
    return result;
}

std::vector<uint8_t> CudaEncryption::decrypt(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key) {
    // XOR encryption is symmetric
    return encrypt(data, key);
}

std::vector<uint8_t> CudaEncryption::encrypt_aes(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key) {
    // Placeholder for more complex AES-like encryption
    // In a real implementation, this would use a proper AES algorithm
    std::cout << "AES encryption not fully implemented (placeholder)" << std::endl;
    return encrypt(data, key);
}

std::vector<uint8_t> CudaEncryption::decrypt_aes(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key) {
    // Placeholder for more complex AES-like decryption
    std::cout << "AES decryption not fully implemented (placeholder)" << std::endl;
    return decrypt(data, key);
}

bool CudaEncryption::is_cuda_available() const {
    return cuda_available_;
}

} // namespace snsdb
