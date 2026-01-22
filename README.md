# SNSDB

Socket Network System Database - A C++ application with support for socket-based web servers, binary file operations, garbage collection, CUDA encryption, and HTTP requests.

## Features

- **Socket Web Server**: Non-HTTP TCP socket server using Boost.Asio
- **Binary File Operations**: Efficient reading and writing of binary data
- **Garbage Collection**: Automatic memory management using Boehm GC
- **CUDA Encryption**: GPU-accelerated encryption algorithms using NVIDIA CUDA
- **HTTP Client**: HTTP requests support using libcurl

## Dependencies

### Required Dependencies

- **CMake** (>= 3.18): Build system
- **C++ Compiler**: Supporting C++17 or later (GCC, Clang, MSVC)

### Optional Dependencies

All dependencies are optional. The build system will detect available libraries and enable corresponding features:

1. **Boost** (>= 1.70) - For socket server functionality
   - Components: system, thread
   - Used for: Non-HTTP TCP socket server with async I/O

2. **libcurl** - For HTTP client functionality
   - Used for: Making HTTP requests (GET, POST, PUT, DELETE)

3. **Boehm GC** - For garbage collection
   - Used for: Automatic memory management

4. **NVIDIA CUDA** (>= 10.0) - For GPU-accelerated encryption
   - Used for: CUDA-accelerated encryption algorithms
   - Requires: NVIDIA GPU with CUDA support

## Building

### Basic Build

```bash
mkdir build
cd build
cmake ..
make
```

### Build Options

Disable CUDA support:
```bash
cmake -DENABLE_CUDA=OFF ..
```

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get install libboost-all-dev libcurl4-openssl-dev libgc-dev
```

For CUDA, follow [NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads).

#### macOS (via Homebrew)
```bash
brew install boost curl bdw-gc
```

For CUDA, install from [NVIDIA CUDA downloads](https://developer.nvidia.com/cuda-downloads).

#### Windows
- Boost: Download from [boost.org](https://www.boost.org/)
- libcurl: Use vcpkg or download from [curl.se](https://curl.se/)
- Boehm GC: Download from [GitHub](https://github.com/ivmai/bdwgc)
- CUDA: Install from [NVIDIA](https://developer.nvidia.com/cuda-downloads)

## Usage

### Running the Demo

```bash
./snsdb
```

### Socket Server Example

```cpp
#include "socket_server.h"

snsdb::SocketServer server(8080);
server.start();  // Starts listening on port 8080
```

### Binary File Operations

```cpp
#include "binary_reader.h"

// Writing binary data
snsdb::BinaryWriter writer("output.bin");
writer.open();
uint32_t value = 0x12345678;
writer.write(value);
writer.close();

// Reading binary data
snsdb::BinaryReader reader("output.bin");
reader.open();
uint32_t read_value;
reader.read(read_value);
reader.close();
```

### Garbage Collection

```cpp
#include "gc_manager.h"

snsdb::GCManager::initialize();

// Allocate memory (automatically garbage collected)
void* ptr = snsdb::GCManager::allocate(1024);

// Force collection
snsdb::GCManager::collect();
```

### CUDA Encryption

```cpp
#include "cuda_encryption.h"

snsdb::CudaEncryption crypto;
std::vector<uint8_t> data = {1, 2, 3, 4, 5};
std::vector<uint8_t> key = {0xAA, 0xBB, 0xCC};

auto encrypted = crypto.encrypt(data, key);
auto decrypted = crypto.decrypt(encrypted, key);
```

### HTTP Client

```cpp
#include "http_client.h"

snsdb::HttpClient client;
client.set_timeout(30);

auto response = client.get("http://example.com");
std::cout << "Status: " << response.status_code << std::endl;
std::cout << "Body: " << response.body << std::endl;
```

## Library Features Summary

| Feature | Library | Purpose |
|---------|---------|---------|
| Socket Server | Boost.Asio | Non-HTTP TCP socket server with async I/O |
| Binary I/O | Standard C++ | Read/write binary files efficiently |
| Garbage Collection | Boehm GC | Automatic memory management |
| Encryption | NVIDIA CUDA | GPU-accelerated encryption algorithms |
| HTTP Requests | libcurl | Make HTTP/HTTPS requests |

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.