#include "socket_server.h"

#ifdef USE_BOOST_ASIO
#include <iostream>
#include <thread>

namespace snsdb {

using boost::asio::ip::tcp;

SocketServer::SocketServer(unsigned short port) 
    : port_(port), running_(false) {
}

SocketServer::~SocketServer() {
    stop();
}

void SocketServer::start() {
    try {
        acceptor_ = std::make_unique<tcp::acceptor>(
            io_context_, 
            tcp::endpoint(tcp::v4(), port_)
        );
        
        running_ = true;
        std::cout << "Socket server started on port " << port_ << std::endl;
        
        accept_connection();
        io_context_.run();
    } catch (std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

void SocketServer::stop() {
    if (running_) {
        running_ = false;
        io_context_.stop();
        if (acceptor_) {
            acceptor_->close();
        }
        std::cout << "Socket server stopped" << std::endl;
    }
}

void SocketServer::accept_connection() {
    auto socket = std::make_shared<tcp::socket>(io_context_);
    
    acceptor_->async_accept(*socket, [this, socket](boost::system::error_code ec) {
        if (!ec) {
            std::cout << "New connection accepted" << std::endl;
            std::thread(&SocketServer::handle_client, this, socket).detach();
        }
        
        if (running_) {
            accept_connection();
        }
    });
}

void SocketServer::handle_client(std::shared_ptr<tcp::socket> socket) {
    try {
        while (true) {
            char data[1024];
            boost::system::error_code error;
            
            size_t length = socket->read_some(boost::asio::buffer(data), error);
            
            if (error == boost::asio::error::eof) {
                std::cout << "Connection closed by client" << std::endl;
                break;
            } else if (error) {
                throw boost::system::system_error(error);
            }
            
            // Echo back the data (simple example)
            boost::asio::write(*socket, boost::asio::buffer(data, length));
        }
    } catch (std::exception& e) {
        std::cerr << "Client handler error: " << e.what() << std::endl;
    }
}

} // namespace snsdb

#endif // USE_BOOST_ASIO
