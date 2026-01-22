#include <iostream>
#include <vector>
#include <string>

// Include all library headers
#include "socket_server.h"
#include "binary_reader.h"
#include "gc_manager.h"
#include "cuda_encryption.h"
#include "http_client.h"

void print_header(const std::string& title) {
    std::cout << "\n========================================" << std::endl;
    std::cout << title << std::endl;
    std::cout << "========================================" << std::endl;
}

void demo_binary_operations() {
    print_header("Binary File Operations Demo");
    
    snsdb::BinaryWriter writer("/tmp/test_binary.dat");
    if (writer.open()) {
        std::cout << "Writing binary data..." << std::endl;
        uint32_t value1 = 0x12345678;
        float value2 = 3.14159f;
        writer.write(value1);
        writer.write(value2);
        
        std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04, 0x05};
        writer.write_bytes(data);
        writer.close();
        std::cout << "Binary data written successfully" << std::endl;
    }
    
    snsdb::BinaryReader reader("/tmp/test_binary.dat");
    if (reader.open()) {
        std::cout << "Reading binary data..." << std::endl;
        uint32_t value1;
        float value2;
        reader.read(value1);
        reader.read(value2);
        
        std::cout << "Read uint32: 0x" << std::hex << value1 << std::dec << std::endl;
        std::cout << "Read float: " << value2 << std::endl;
        
        std::vector<uint8_t> data = reader.read_all();
        std::cout << "Total file size: " << data.size() << " bytes" << std::endl;
        reader.close();
    }
}

void demo_gc() {
    print_header("Garbage Collection Demo");
    
    snsdb::GCManager::initialize();
    
    // Allocate some memory
    void* ptr1 = snsdb::GCManager::allocate(1024);
    void* ptr2 = snsdb::GCManager::allocate(2048);
    
    std::cout << "Allocated memory blocks" << std::endl;
    std::cout << "Heap size: " << snsdb::GCManager::get_heap_size() << " bytes" << std::endl;
    std::cout << "Free bytes: " << snsdb::GCManager::get_free_bytes() << " bytes" << std::endl;
    
    // Force garbage collection
    snsdb::GCManager::collect();
    std::cout << "Garbage collection completed" << std::endl;
}

void demo_encryption() {
    print_header("CUDA Encryption Demo");
    
    snsdb::CudaEncryption crypto;
    
    std::string message = "Hello, SNSDB with CUDA encryption!";
    std::vector<uint8_t> data(message.begin(), message.end());
    std::vector<uint8_t> key = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    
    std::cout << "Original message: " << message << std::endl;
    std::cout << "CUDA available: " << (crypto.is_cuda_available() ? "Yes" : "No") << std::endl;
    
    // Encrypt
    auto encrypted = crypto.encrypt(data, key);
    std::cout << "Encrypted data size: " << encrypted.size() << " bytes" << std::endl;
    
    // Decrypt
    auto decrypted = crypto.decrypt(encrypted, key);
    std::string decrypted_msg(decrypted.begin(), decrypted.end());
    std::cout << "Decrypted message: " << decrypted_msg << std::endl;
}

void demo_http_client() {
    print_header("HTTP Client Demo");
    
#ifdef USE_CURL
    snsdb::HttpClient client;
    client.set_timeout(10);
    
    std::cout << "Making HTTP GET request to httpbin.org..." << std::endl;
    auto response = client.get("http://httpbin.org/get");
    
    std::cout << "Status code: " << response.status_code << std::endl;
    if (response.status_code == 200) {
        std::cout << "Response body (first 200 chars): " 
                  << response.body.substr(0, 200) << "..." << std::endl;
    }
#else
    std::cout << "HTTP client not available (libcurl not found)" << std::endl;
#endif
}

void demo_socket_server() {
    print_header("Socket Server Demo");
    
#ifdef USE_BOOST_ASIO
    std::cout << "Socket server functionality available" << std::endl;
    std::cout << "To start server, uncomment the code below and run:" << std::endl;
    std::cout << "  snsdb::SocketServer server(8080);" << std::endl;
    std::cout << "  server.start();" << std::endl;
    std::cout << "\nNote: Server is not started in demo mode to avoid blocking" << std::endl;
    
    // Uncomment to actually start the server
    // snsdb::SocketServer server(8080);
    // server.start();
#else
    std::cout << "Socket server not available (Boost.Asio not found)" << std::endl;
#endif
}

int main(int argc, char* argv[]) {
    std::cout << "SNSDB - Socket Network System Database" << std::endl;
    std::cout << "Library demonstration program" << std::endl;
    
    // Run all demos
    demo_binary_operations();
    demo_gc();
    demo_encryption();
    demo_http_client();
    demo_socket_server();
    
    print_header("Demo Complete");
    std::cout << "All library features demonstrated." << std::endl;
    
    return 0;
}
