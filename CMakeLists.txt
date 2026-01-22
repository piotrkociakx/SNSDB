cmake_minimum_required(VERSION 3.18)
project(SNSDB VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA support for encryption algorithms
option(ENABLE_CUDA "Enable NVIDIA CUDA support for encryption" ON)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
            set(CMAKE_CUDA_STANDARD 14)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            add_definitions(-DUSE_CUDA)
        else()
            message(STATUS "CUDA toolkit not found, building without CUDA support")
            set(ENABLE_CUDA OFF)
        endif()
    else()
        message(STATUS "CUDA compiler not found, building without CUDA support")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Find Boost for Asio (socket web server)
find_package(Boost 1.70 COMPONENTS system thread QUIET)
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    add_definitions(-DUSE_BOOST_ASIO)
else()
    message(STATUS "Boost not found, socket server will not be available")
endif()

# Find libcurl for HTTP requests
find_package(CURL QUIET)
if(CURL_FOUND)
    message(STATUS "libcurl found: ${CURL_VERSION_STRING}")
    add_definitions(-DUSE_CURL)
else()
    message(STATUS "libcurl not found, HTTP requests will not be available")
endif()

# Find Boehm GC for garbage collection
find_library(GC_LIBRARY NAMES gc)
find_path(GC_INCLUDE_DIR gc.h)
if(GC_LIBRARY AND GC_INCLUDE_DIR)
    message(STATUS "Boehm GC found")
    add_definitions(-DUSE_GC)
else()
    message(STATUS "Boehm GC not found, manual memory management will be used")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

if(CURL_FOUND)
    include_directories(${CURL_INCLUDE_DIRS})
endif()

if(GC_INCLUDE_DIR)
    include_directories(${GC_INCLUDE_DIR})
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/binary_reader.cpp
    src/gc_manager.cpp
    src/http_client.cpp
)

# Conditionally add socket server source
if(Boost_FOUND)
    list(APPEND SOURCES src/socket_server.cpp)
endif()

# Conditionally add CUDA encryption source (non-CUDA version)
if(NOT ENABLE_CUDA OR NOT CUDAToolkit_FOUND)
    list(APPEND SOURCES src/cuda_encryption.cpp)
endif()

# Create executable
add_executable(snsdb ${SOURCES})

# Link libraries
if(Boost_FOUND)
    target_link_libraries(snsdb ${Boost_LIBRARIES} pthread)
endif()

if(CURL_FOUND)
    target_link_libraries(snsdb ${CURL_LIBRARIES})
endif()

if(GC_LIBRARY)
    target_link_libraries(snsdb ${GC_LIBRARY})
endif()

# CUDA encryption module (optional)
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    set(CUDA_SOURCES
        src/cuda_encryption.cu
    )
    add_library(cuda_encryption ${CUDA_SOURCES})
    target_link_libraries(snsdb cuda_encryption)
endif()

# Installation
install(TARGETS snsdb DESTINATION bin)
