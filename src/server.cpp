#include "server.hpp"

#include <uWebSockets/App.h> // Web sockets

#include <atomic>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp> // JSON library for parsing and generating JSON data
#include <string>
#include <thread>

#include <Camera.hpp>
#include "ThreadPool.hpp"

// Global variables
static ThreadPool pool(1);
static uWS::App *globalApp; // Pointer to uWebSockets application instance
static std::atomic<bool> isStreaming{false};
static AICamera camera;

/**
 * Callback function to handle new frames from the camera.
 * THis function is set as the frame callback in AICamera.
 */
static void handleNewFrame(
  const std::vector<unsigned char> &buffer,
  const std::vector<HailoDetectionPtr> &detections
)
{
  // Get the event loop from the global uWebSockets app
  uWS::Loop *loop = globalApp->getLoop();

  // Defer the following lambda to be executed in the event loop.
  // The lambda converts the frame buffer to a string_view and publishes it as binary data
  // to the "image-stream" channel
  loop->defer(
    [buffer]()
    {
      // Create a string view from the raw buffer data
      std::string_view stringView = std::string_view((char *)buffer.data(), buffer.size());
      // Set the opcode for binary data
      uWS::OpCode opCode = uWS::BINARY;
      // Pulish the frame data on the "image-stream" channel
      globalApp->publish("image-stream", stringView, opCode);
    }
  );
}

/**
 * Template function to log HTTP requests.
 * This function logs the method and URL of incoming HTTP requests.
 */
template <bool SSL>
void logRequest(uWS::HttpResponse<SSL> *res, uWS::HttpRequest *req)
{
  std::cout << "Incoming request: " << req->getMethod() << " " << req->getUrl() << std::endl;
}

/**
 * Function to start the web server.
 * This function creates the uWebSockets app, sets up HTTP routes and WebSocket endpoints,
 * configures the camera, and then starts the event loop.
 */
void startServer(int port = 9898)
{
  // Note that SSL is disabled unless you build with WITH_OPENSSL=1
  globalApp = new uWS::App();
  uWS::App &app = *globalApp;

  // Set the camera's frame callback to our handleNewFrame function
  camera.setFrameReadyCallback(handleNewFrame);

  app
    .post(
      "/start",
      [](auto *res, auto *req)
      {
        // Enqueue a task in the thread pool that starts and runs the camera capture pipeline
        pool.enqueue(
          []()
          {
            camera.startCameraPipeline();
            camera.startFrameCapture();
          }
        );

        isStreaming = true;

        nlohmann::json responseJson = {{ "message", "stream started"}};
        std::string response = responseJson.dump();
        res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
        logRequest(res, req);
      }
    )
    .post(
      "/stop",
      [](auto *res, auto *req)
      {
        // Stop the camera capture pipeline
        camera.finish();
        // Respond with a JSON message indicating that streaming has stopped
        nlohmann::json responseJson = {{ "message", "stream stopped" }};
        std::string response = responseJson.dump();
        res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
        logRequest(res, req);
      }
    )
    .ws<std::string>(
      "/ws",
      {
        .compression = uWS::SHARED_COMPRESSOR,
        .maxPayloadLength = 16 * 1024 * 1024, // 16MB
        .idleTimeout = 10,
        .open =
          [](auto *ws)
          {
            // When a new WebSocket connection opens, log the connection
            std::cout << "Thread: " << std::this_thread::get_id() << " connected" << std::endl;
            ws->subscribe("image-stream");
          },
        .message =
          [](auto *ws, std::string_view message, uWS::OpCode opCode)
          {
            // For incoming messages, subscribe to "image-stream" (or handle messages as needed)
            ws->subscribe("image-stream");
          },
        .close =
          [](auto *ws, int code, std::string_view message)
          {
            // Log when a WebSocket connection is closed
            std::cout << "Thread: " << std::this_thread::get_id() << " disconnected" << std::endl;
          }
      }
    )
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

  // When the event loop ends, finish the camera capture pipeline
  camera.finish();

  std::cout << "Thread: " << std::this_thread::get_id() << " server ended" << std::endl;
}