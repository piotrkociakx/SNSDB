#ifndef GC_MANAGER_H
#define GC_MANAGER_H

#ifdef USE_GC
#include <gc.h>
#endif

#include <memory>
#include <cstddef>

namespace snsdb {

/**
 * @brief Garbage collection manager using Boehm GC
 * 
 * Provides automatic memory management capabilities.
 */
class GCManager {
public:
    static void initialize();
    static void collect();
    static void* allocate(size_t size);
    static void deallocate(void* ptr);
    
    // Statistics
    static size_t get_heap_size();
    static size_t get_free_bytes();
    static size_t get_total_bytes();
};

/**
 * @brief RAII wrapper for GC-allocated memory
 */
template<typename T>
class GCPtr {
public:
    GCPtr() : ptr_(nullptr) {}
    explicit GCPtr(T* ptr) : ptr_(ptr) {}
    
    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    
    operator bool() const { return ptr_ != nullptr; }
    
private:
    T* ptr_;
};

} // namespace snsdb

#endif // GC_MANAGER_H
