#include "gc_manager.h"
#include <iostream>

namespace snsdb {

void GCManager::initialize() {
#ifdef USE_GC
    GC_INIT();
    std::cout << "Garbage collector initialized" << std::endl;
#else
    std::cout << "Garbage collector not available (USE_GC not defined)" << std::endl;
#endif
}

void GCManager::collect() {
#ifdef USE_GC
    GC_gcollect();
#endif
}

void* GCManager::allocate(size_t size) {
#ifdef USE_GC
    return GC_MALLOC(size);
#else
    return malloc(size);
#endif
}

void GCManager::deallocate(void* ptr) {
#ifdef USE_GC
    // With GC, explicit deallocation is not needed
    // GC_FREE(ptr); // Optional
#else
    free(ptr);
#endif
}

size_t GCManager::get_heap_size() {
#ifdef USE_GC
    return GC_get_heap_size();
#else
    return 0;
#endif
}

size_t GCManager::get_free_bytes() {
#ifdef USE_GC
    return GC_get_free_bytes();
#else
    return 0;
#endif
}

size_t GCManager::get_total_bytes() {
#ifdef USE_GC
    return GC_get_total_bytes();
#else
    return 0;
#endif
}

} // namespace snsdb
