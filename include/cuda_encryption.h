#ifndef CUDA_ENCRYPTION_H
#define CUDA_ENCRYPTION_H

#include <vector>
#include <cstdint>

namespace snsdb {

/**
 * @brief CUDA-accelerated encryption algorithms
 * 
 * Provides GPU-accelerated encryption/decryption using NVIDIA CUDA.
 */
class CudaEncryption {
public:
    CudaEncryption();
    ~CudaEncryption();
    
    // XOR-based encryption (simple example)
    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key);
    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key);
    
    // AES-like encryption (placeholder for more complex algorithm)
    std::vector<uint8_t> encrypt_aes(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key);
    std::vector<uint8_t> decrypt_aes(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key);
    
    bool is_cuda_available() const;
    
private:
    bool cuda_available_;
};

} // namespace snsdb

#endif // CUDA_ENCRYPTION_H
