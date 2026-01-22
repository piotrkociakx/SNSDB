#ifndef BINARY_READER_H
#define BINARY_READER_H

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

namespace snsdb {

/**
 * @brief Binary file reader utility
 * 
 * Provides methods for reading binary data from files efficiently.
 */
class BinaryReader {
public:
    explicit BinaryReader(const std::string& filename);
    ~BinaryReader();
    
    bool open();
    void close();
    bool is_open() const;
    
    // Read various data types
    template<typename T>
    bool read(T& value);
    
    bool read_bytes(uint8_t* buffer, size_t size);
    std::vector<uint8_t> read_all();
    bool seek(size_t position);
    size_t tell() const;
    size_t size() const;
    
private:
    std::string filename_;
    std::ifstream file_;
    size_t file_size_;
};

/**
 * @brief Binary file writer utility
 */
class BinaryWriter {
public:
    explicit BinaryWriter(const std::string& filename);
    ~BinaryWriter();
    
    bool open();
    void close();
    bool is_open() const;
    
    // Write various data types
    template<typename T>
    bool write(const T& value);
    
    bool write_bytes(const uint8_t* buffer, size_t size);
    bool write_bytes(const std::vector<uint8_t>& data);
    
private:
    std::string filename_;
    std::ofstream file_;
};

} // namespace snsdb

#endif // BINARY_READER_H
