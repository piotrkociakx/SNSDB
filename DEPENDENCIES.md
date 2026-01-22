# SNSDB Dependencies and Requirements

## Overview
This document lists all the libraries and dependencies used in SNSDB.

## Core Dependencies

### 1. Socket Web Server (Non-HTTP)
**Library**: Boost.Asio
- **Version**: >= 1.70
- **Purpose**: Provides asynchronous I/O capabilities for building a TCP socket server
- **Components**: boost_system, boost_thread
- **Features**:
  - Non-blocking socket operations
  - Cross-platform socket abstraction
  - Async connection handling
  - Thread-safe operations

### 2. Binary File Reading and Writing
**Library**: Standard C++ (fstream)
- **Version**: C++17 standard library
- **Purpose**: Efficient binary file I/O operations
- **Features**:
  - Type-safe binary reading/writing
  - Random access to file contents
  - Support for all primitive types
  - Zero-copy operations where possible

### 3. Garbage Collection
**Library**: Boehm-Demers-Weiser Garbage Collector (libgc)
- **Version**: >= 7.6
- **Purpose**: Automatic memory management
- **Features**:
  - Conservative garbage collection
  - Mark-and-sweep algorithm
  - Thread-safe allocation
  - Memory leak detection
  - Minimal runtime overhead

### 4. NVIDIA CUDA for Encryption
**Library**: NVIDIA CUDA Toolkit
- **Version**: >= 10.0
- **Purpose**: GPU-accelerated encryption algorithms
- **Features**:
  - Parallel encryption/decryption
  - XOR-based encryption (example)
  - Support for custom encryption algorithms
  - Automatic CPU fallback when GPU unavailable
  - Significant performance boost for large datasets

### 5. HTTP Requests
**Library**: libcurl
- **Version**: >= 7.58
- **Purpose**: HTTP/HTTPS client functionality
- **Features**:
  - Support for all HTTP methods (GET, POST, PUT, DELETE)
  - HTTPS support with SSL/TLS
  - Custom headers
  - Binary data download
  - Timeout configuration
  - Proxy support

### 6. Binary File Reading (Additional)
**Library**: Standard C++ (fstream, vector)
- **Version**: C++17 standard library
- **Purpose**: Read binary files into memory
- **Features**:
  - Read entire files into memory
  - Stream-based reading
  - Random access
  - Efficient for large files

## Build System

### CMake
- **Version**: >= 3.18
- **Purpose**: Cross-platform build system configuration
- **Features**:
  - Automatic dependency detection
  - Conditional compilation based on available libraries
  - Support for multiple platforms (Linux, macOS, Windows)
  - CUDA integration

## Optional Features

All dependencies except the C++ standard library are optional. The build system will:
- Detect available libraries
- Enable corresponding features
- Provide graceful degradation when libraries are missing
- Display clear messages about available/unavailable features

## Platform-Specific Notes

### Linux
- Most distributions provide packages for all dependencies
- CUDA requires NVIDIA driver installation

### macOS
- All dependencies available via Homebrew
- CUDA only available on Macs with NVIDIA GPUs (older models)

### Windows
- Dependencies available via vcpkg or manual installation
- CUDA requires NVIDIA GPU and drivers
- Visual Studio recommended for building

## Installation Commands

### Ubuntu/Debian
```bash
sudo apt-get install cmake g++ libboost-all-dev libcurl4-openssl-dev libgc-dev
```

### Fedora/RHEL
```bash
sudo dnf install cmake gcc-c++ boost-devel libcurl-devel gc-devel
```

### macOS
```bash
brew install cmake boost curl bdw-gc
```

### Windows (vcpkg)
```bash
vcpkg install boost-asio boost-system boost-thread curl libgc
```

## Verification

After installing dependencies, verify with:
```bash
cmake --version          # Should be >= 3.18
gcc --version           # Or your C++ compiler
pkg-config --list-all | grep -E "(boost|curl|gc)"
```

## Security Considerations

- libcurl: Always use HTTPS for sensitive data
- CUDA encryption: Example implementations, not production-ready
- Boehm GC: May have security implications in certain contexts
- Keep all libraries updated to latest stable versions
