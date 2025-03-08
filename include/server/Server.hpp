#ifndef SERVER_HPP
#define SERVER_HPP

#include "../ThreadPool.hpp"
#include <uWebSockets/App.h>
#include <atomic>
#include "../Camera.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>

/**
 * The Server class encapsulates the functionality to start a web server and stream video frames.
 * Only one instance should be created so the class implements the singleton design pattern.
 */
class Server
{
  private:
    std::mutex io_mutex; // For locking I/O operations
    ThreadPool pool;
    uWS::App app;
    std::atomic<bool> isStreaming{false};
    Camera camera;

    explicit Server() : pool(1), app()
    {
      camera.setFrameReadyCallback(
        // Use lambda to capture "this" pointer.
        [this](const std::vector<unsigned char> &buffer, const std::vector<HailoDetectionPtr> &detections)
        {
          this->handleNewFrame(buffer, detections);
        }
      );
      this->setupRoutes();
    }

    /**
     * Callback function to handle frames after inference has been performed.
     * 
     * @param buffer The frame buffer.
     * @param detections The detections found in the frame.
     */
    void handleNewFrame(const std::vector<unsigned char> &buffer, const std::vector<HailoDetectionPtr> &detections)
    {
      uWS::Loop *loop = this->app.getLoop(); // Event loop

      if (loop == nullptr) {
        std::cerr << "Error: Server::handleNewFrame: Event loop is null" << std::endl;
        return;
      }

      loop->defer(
        [this, buffer]()
        {
          std::string_view string_view = std::string_view((char *)buffer.data(), buffer.size());
          uWS::OpCode op_code = uWS::BINARY;
          this->app.publish("image-stream", string_view, op_code);
        }
      );
    }

    /**
     * Log HTTP requests.
     * 
     * @param res THe HTTP response object.
     * @param req The HTTP request object.
     */
    template <bool SSL>
    void logRequest(uWS::HttpResponse<SSL> *res, uWS::HttpRequest *req)
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cout << "Incoming request: " << req->getMethod() << " " << req->getUrl() << std::endl;
    }

    /**
     * Set up API routes.
     */
    void setupRoutes()
    {
      this->app
        .post(
          "/start",
          [this](auto *res, auto *req)
          {
            this->pool.enqueue(
              [this]()
              {
                this->camera.startCameraPipeline();
                this->camera.startFrameCapture();
              }
            );

            this->isStreaming = true;

            nlohmann::json responseJson = {{ "message", "stream started"}};
            std::string response = responseJson.dump();
            res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
            this->logRequest(res, req);
          }
        )
        .post(
          "/stop",
          [this](auto *res, auto *req)
          {
            this->camera.finish(); // Stop the camera capture pipeline
            this->isStreaming = false;
            nlohmann::json responseJson = {{ "message", "stream stopped" }};
            std::string response = responseJson.dump();
            res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
            this->logRequest(res, req);
          }
        )
        .ws<std::string>(
          "/ws",
          {
            .compression = uWS::SHARED_COMPRESSOR,
            .maxPayloadLength = 16 * 1024 * 1024, // 16MB
            .idleTimeout = 10,
            .open =
              [this](auto *ws)
              {
                std::lock_guard<std::mutex> lock(this->io_mutex);
                std::cout << "Thread: " << std::this_thread::get_id() << " connected" << std::endl;
                ws->subscribe("image-stream");
              },
            .message =
              [](auto *ws, std::string_view message, uWS::OpCode op_code)
              {
                ws->subscribe("image-stream");
              },
            .close =
              [this](auto *ws, int code, std::string_view message)
              {
                std::lock_guard<std::mutex> lock(this->io_mutex);
                std::cout << "Thread: " << std::this_thread::get_id() << " disconnected" << std::endl;
              }
          }
        );
    }

  public:
    // Delete copy constructor and assignment operator to prevent copying
    Server(const Server &) = delete;
    Server& operator=(const Server &) = delete;
    
    /**
     * Static method to get the singleton instance.
     */
    static Server& getInstance()
    {
      static Server instance;
      return instance;
    }

    /**
     * Start the web server.
     * 
     * @param port The port to listen on.
     */
    void startServer(int port = 9898)
    {
      this->app
        .listen(
          port,
          [](auto *listen_socket)
          {
            if (listen_socket) {
              // us_socket_local_port retrieves the port number from the listen socket
              std::cout << "Thread " << std::this_thread::get_id() << " listening on port ";
              std::cout << us_socket_local_port(true, (struct us_socket_t *)listen_socket) << std::endl;
            }
            else {
              std::cout << "Thread: " << std::this_thread::get_id() << " failed to listen on port 9898" << std::endl;
            }
          }
        )
        .run();

      // When the event loop finishes, stop the camera
      this->camera.finish();
      std::cout << "Thread: " << std::this_thread::get_id() << " server ended" << std::endl;
    }

    ~Server()
    {
      this->camera.finish();
    }
};

#endif  // SERVER_HPP