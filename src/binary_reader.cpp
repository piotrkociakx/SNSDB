#include "binary_reader.h"
#include <stdexcept>
#include <cstring>

namespace snsdb {

// BinaryReader implementation
BinaryReader::BinaryReader(const std::string& filename)
    : filename_(filename), file_size_(0) {
}

BinaryReader::~BinaryReader() {
    close();
}

bool BinaryReader::open() {
    file_.open(filename_, std::ios::binary | std::ios::in);
    if (!file_.is_open()) {
        return false;
    }
    
    // Get file size
    file_.seekg(0, std::ios::end);
    file_size_ = file_.tellg();
    file_.seekg(0, std::ios::beg);
    
    return true;
}

void BinaryReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool BinaryReader::is_open() const {
    return file_.is_open();
}

template<typename T>
bool BinaryReader::read(T& value) {
    if (!is_open()) return false;
    file_.read(reinterpret_cast<char*>(&value), sizeof(T));
    return file_.good();
}

bool BinaryReader::read_bytes(uint8_t* buffer, size_t size) {
    if (!is_open() || !buffer) return false;
    file_.read(reinterpret_cast<char*>(buffer), size);
    return file_.good();
}

std::vector<uint8_t> BinaryReader::read_all() {
    if (!is_open()) return std::vector<uint8_t>();
    
    std::vector<uint8_t> data(file_size_);
    file_.seekg(0, std::ios::beg);
    file_.read(reinterpret_cast<char*>(data.data()), file_size_);
    
    return data;
}

bool BinaryReader::seek(size_t position) {
    if (!is_open()) return false;
    file_.seekg(position, std::ios::beg);
    return file_.good();
}

size_t BinaryReader::tell() const {
    if (!is_open()) return 0;
    return const_cast<std::ifstream&>(file_).tellg();
}

size_t BinaryReader::size() const {
    return file_size_;
}

// BinaryWriter implementation
BinaryWriter::BinaryWriter(const std::string& filename)
    : filename_(filename) {
}

BinaryWriter::~BinaryWriter() {
    close();
}

bool BinaryWriter::open() {
    file_.open(filename_, std::ios::binary | std::ios::out);
    return file_.is_open();
}

void BinaryWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool BinaryWriter::is_open() const {
    return file_.is_open();
}

template<typename T>
bool BinaryWriter::write(const T& value) {
    if (!is_open()) return false;
    file_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return file_.good();
}

bool BinaryWriter::write_bytes(const uint8_t* buffer, size_t size) {
    if (!is_open() || !buffer) return false;
    file_.write(reinterpret_cast<const char*>(buffer), size);
    return file_.good();
}

bool BinaryWriter::write_bytes(const std::vector<uint8_t>& data) {
    return write_bytes(data.data(), data.size());
}

// Explicit template instantiations
template bool BinaryReader::read<uint8_t>(uint8_t&);
template bool BinaryReader::read<uint16_t>(uint16_t&);
template bool BinaryReader::read<uint32_t>(uint32_t&);
template bool BinaryReader::read<uint64_t>(uint64_t&);
template bool BinaryReader::read<int8_t>(int8_t&);
template bool BinaryReader::read<int16_t>(int16_t&);
template bool BinaryReader::read<int32_t>(int32_t&);
template bool BinaryReader::read<int64_t>(int64_t&);
template bool BinaryReader::read<float>(float&);
template bool BinaryReader::read<double>(double&);

template bool BinaryWriter::write<uint8_t>(const uint8_t&);
template bool BinaryWriter::write<uint16_t>(const uint16_t&);
template bool BinaryWriter::write<uint32_t>(const uint32_t&);
template bool BinaryWriter::write<uint64_t>(const uint64_t&);
template bool BinaryWriter::write<int8_t>(const int8_t&);
template bool BinaryWriter::write<int16_t>(const int16_t&);
template bool BinaryWriter::write<int32_t>(const int32_t&);
template bool BinaryWriter::write<int64_t>(const int64_t&);
template bool BinaryWriter::write<float>(const float&);
template bool BinaryWriter::write<double>(const double&);

} // namespace snsdb
