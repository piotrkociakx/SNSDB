#ifndef SOCKET_SERVER_H
#define SOCKET_SERVER_H

#ifdef USE_BOOST_ASIO
#include <boost/asio.hpp>
#include <memory>
#include <string>

namespace snsdb {

/**
 * @brief A simple socket-based (non-HTTP) web server using Boost.Asio
 * 
 * This class provides a TCP socket server for handling custom binary protocols.
 */
class SocketServer {
public:
    SocketServer(unsigned short port);
    ~SocketServer();
    
    void start();
    void stop();
    
private:
    void accept_connection();
    void handle_client(std::shared_ptr<boost::asio::ip::tcp::socket> socket);
    
    boost::asio::io_context io_context_;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_;
    unsigned short port_;
    bool running_;
};

} // namespace snsdb

#endif // USE_BOOST_ASIO
#endif // SOCKET_SERVER_H
