#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

#ifdef USE_CURL
#include <curl/curl.h>
#endif

#include <string>
#include <map>
#include <vector>
#include <cstdint>

namespace snsdb {

/**
 * @brief HTTP client for making requests using libcurl
 */
class HttpClient {
public:
    HttpClient();
    ~HttpClient();
    
    struct Response {
        int status_code;
        std::string body;
        std::map<std::string, std::string> headers;
    };
    
    // HTTP methods
    Response get(const std::string& url);
    Response post(const std::string& url, const std::string& data);
    Response put(const std::string& url, const std::string& data);
    Response del(const std::string& url);
    
    // Download binary data
    std::vector<uint8_t> download_binary(const std::string& url);
    
    // Set custom headers
    void set_header(const std::string& key, const std::string& value);
    void clear_headers();
    
    // Timeout settings
    void set_timeout(long seconds);
    
private:
#ifdef USE_CURL
    CURL* curl_;
#endif
    std::map<std::string, std::string> headers_;
    long timeout_;
    
    Response perform_request(const std::string& url, const std::string& method, const std::string& data = "");
};

} // namespace snsdb

#endif // HTTP_CLIENT_H
