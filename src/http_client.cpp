#include "http_client.h"
#include <iostream>
#include <cstring>

namespace snsdb {

#ifdef USE_CURL
// Callback function for libcurl to write response data
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static size_t write_binary_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::vector<uint8_t>* vec = (std::vector<uint8_t>*)userp;
    uint8_t* data = (uint8_t*)contents;
    vec->insert(vec->end(), data, data + (size * nmemb));
    return size * nmemb;
}
#endif

HttpClient::HttpClient() : timeout_(30) {
#ifdef USE_CURL
    curl_ = curl_easy_init();
    if (!curl_) {
        std::cerr << "Failed to initialize CURL" << std::endl;
    }
#else
    std::cout << "HTTP client not available (USE_CURL not defined)" << std::endl;
#endif
}

HttpClient::~HttpClient() {
#ifdef USE_CURL
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
#endif
}

HttpClient::Response HttpClient::get(const std::string& url) {
    return perform_request(url, "GET");
}

HttpClient::Response HttpClient::post(const std::string& url, const std::string& data) {
    return perform_request(url, "POST", data);
}

HttpClient::Response HttpClient::put(const std::string& url, const std::string& data) {
    return perform_request(url, "PUT", data);
}

HttpClient::Response HttpClient::del(const std::string& url) {
    return perform_request(url, "DELETE");
}

std::vector<uint8_t> HttpClient::download_binary(const std::string& url) {
    std::vector<uint8_t> result;
    
#ifdef USE_CURL
    if (!curl_) return result;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_binary_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &result);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, timeout_);
    
    CURLcode res = curl_easy_perform(curl_);
    if (res != CURLE_OK) {
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        result.clear();
    }
#endif
    
    return result;
}

void HttpClient::set_header(const std::string& key, const std::string& value) {
    headers_[key] = value;
}

void HttpClient::clear_headers() {
    headers_.clear();
}

void HttpClient::set_timeout(long seconds) {
    timeout_ = seconds;
}

HttpClient::Response HttpClient::perform_request(const std::string& url, const std::string& method, const std::string& data) {
    Response response;
    response.status_code = 0;
    
#ifdef USE_CURL
    if (!curl_) return response;
    
    std::string response_body;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, timeout_);
    
    // Set method
    if (method == "POST") {
        curl_easy_setopt(curl_, CURLOPT_POST, 1L);
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
    } else if (method == "PUT") {
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "PUT");
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
    } else if (method == "DELETE") {
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
    } else {
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
    }
    
    // Set headers
    struct curl_slist* header_list = nullptr;
    for (const auto& header : headers_) {
        std::string header_str = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, header_str.c_str());
    }
    if (header_list) {
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
    }
    
    // Perform request
    CURLcode res = curl_easy_perform(curl_);
    
    if (res == CURLE_OK) {
        long status_code;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &status_code);
        response.status_code = static_cast<int>(status_code);
        response.body = response_body;
    } else {
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
    }
    
    if (header_list) {
        curl_slist_free_all(header_list);
    }
#else
    std::cout << "HTTP request not available (USE_CURL not defined)" << std::endl;
#endif
    
    return response;
}

} // namespace snsdb
