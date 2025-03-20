#pragma once

#include "global/log_colours.hpp"
#include "global/io_mutex.hpp"
#include "../ThreadPool.hpp"
#include <uWebSockets/App.h>
#include <atomic>
#include "../Camera.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include "HardwareMonitor.hpp"

using Buffer = std::vector<uint8_t>;

/**
 * The Server class encapsulates the functionality to start a web server and stream video frames.
 * Only one instance should be created so the class implements the singleton design pattern.
 */
class Server
{
  private:
    std::mutex startStopMutex;

    std::chrono::time_point<std::chrono::system_clock> lastStartTime;

    ThreadPool threadPool;
    uWS::App app;
    uWS::Loop *loop;
    std::atomic<bool> isApplicationStarted{false};
    std::atomic<bool> isServerStarted{false};
    Camera &camera;
    DetectionInference &detectionInference;

    std::unordered_set<uWS::WebSocket<false, true, std::string> *> clients;
    std::mutex clientsMutex;
    std::chrono::time_point<std::chrono::system_clock> lastPersonNotificationTime;

    HardwareMonitor hardwareMonitor;
    CPUStats previousCpuStats{0};

    std::thread hailoDeviceTemperatureThread;
    std::thread cpuThread;
    std::thread ramThread;
    std::thread applicationStatusThread;

    const uint8_t jpegFrameQuality;
    std::vector<int> jpegCompressionParams;

    Server(const uint8_t jpeg_frame_quality);

    /**
     * Convert a frame to JPEG format and then handle the frame.
     * 
     * @param frame The OpenCV frame to convert.
     * @param detections The detections found in the frame.
     */
    std::optional<Buffer> convertFrameToJpeg(cv::Mat &frame) const;

    /**
     * Broadcast a notification to all connected clients.
     * 
     * @param message The message to send.
     */
    void broadcastNotification(const std::string &message);

    /**
     * Send a notification to all connected clients with the CPU stats.
     */
    void sendCpuStatsNotification();

    /**
     * Send a notification to all connected clients with the Hailo device temperature.
     */
    void sendHailoDeviceTemperatureNotification();

    /**
     * Send a notification to all connected clients with the application status.
     */
    void sendApplicationStatus();

    /**
     * Send a notification to all connected clients with the RAM stats.
     */
    void sendRamNotification();

    /**
     * Callback function to handle frames after inference has been performed.
     * 
     * @param frame The OpenCV frame.
     * @param detections The detections found in the frame.
     */
    void handleNewFrame(cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections);

    /**
     * Log HTTP requests.
     * 
     * @param res THe HTTP response object.
     * @param req The HTTP request object.
     */
    template <bool SSL>
    void logRequest(uWS::HttpResponse<SSL> *res, uWS::HttpRequest *req) const;

    /**
     * Set up API routes.
     */
    void setupRoutes();

  public:
    // Delete copy constructor and assignment operator to prevent copying
    Server(const Server &) = delete;
    Server& operator=(const Server &) = delete;
    
    /**
     * Static method to get the singleton instance.
     * If the server instance does not exist, it will be created and returned.
     */
    static Server& getInstance(const uint8_t jpeg_frame_quality);

    /**
     * Start the web server.
     * 
     * @param port The port to listen on.
     */
    void startServer(int port);

    /**
     * Gracefully destroy the Server instance.
     */
    ~Server();
};

Server::Server(const uint8_t jpeg_frame_quality)
  : threadPool(4),
    loop(this->app.getLoop()),
    lastStartTime(std::chrono::high_resolution_clock::now()),
    lastPersonNotificationTime(std::chrono::high_resolution_clock::now()),
    camera(Camera::getInstance()),
    detectionInference(DetectionInference::getInstance()),
    jpegFrameQuality(jpeg_frame_quality),
    jpegCompressionParams{cv::IMWRITE_JPEG_QUALITY, jpeg_frame_quality, cv::IMWRITE_JPEG_OPTIMIZE, 1}
{
  detectionInference.setFrameReadyCallback(
    // Use lambda to capture "this" pointer.
    [this](cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections)
    {
      this->handleNewFrame(frame, detections);
    }
  );
  this->setupRoutes();

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDMAGENTA << "\nServer setup complete" << std::endl << RESET;
  }
}

std::optional<Buffer> Server::convertFrameToJpeg(cv::Mat &frame) const
{
  Buffer jpeg_buffer;

  // Convert frame to JPEG format and store in jpeg_buffer
  if (!cv::imencode(".jpg", frame, jpeg_buffer, this->jpegCompressionParams)) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDYELLOW << "Error in Server:convertFrameToJpeg:" << std::endl;
    std::cout << "Could not encode the image to JPEG format." << std::endl << RESET;
    return std::nullopt;
  }

  return jpeg_buffer;
}

void Server::broadcastNotification(const std::string &message)
{
  std::lock_guard<std::mutex> lock(this->clientsMutex);
  for (auto *ws : this->clients) {
    this->loop->defer([ws, message] { ws->send(message, uWS::OpCode::TEXT); });
  }
}

void Server::sendCpuStatsNotification()
{
  while (this->isApplicationStarted) {
    const CPUStats current_stats = this->hardwareMonitor.updateCpuStats();
    const double cpu_usage = HardwareMonitor::calculateCpuUsage(this->previousCpuStats, current_stats);
    this->previousCpuStats = current_stats;

    const nlohmann::json cpu_usage_notification = {
      { "event", "cpu stats" },
      { "usage", cpu_usage },
      { "temperature", this->previousCpuStats.temperature }
    };
    const std::string cpu_usage_notification_str = cpu_usage_notification.dump();
    this->broadcastNotification(cpu_usage_notification_str);

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void Server::sendHailoDeviceTemperatureNotification()
{
  while (this->isApplicationStarted) {
    const std::optional<float32_t> device_temp = this->detectionInference.getHailoDeviceTemp();

    if (device_temp) {
      const nlohmann::json temp_notification = {{ "event", "hailo device temperature"}, { "temperature", *device_temp }};
      const std::string temp_notification_str = temp_notification.dump();
      this->broadcastNotification(temp_notification_str);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void Server::sendApplicationStatus()
{
  while (this->isServerStarted) {
    const nlohmann::json application_status_json = {
      { "message", std::string("application ") + (this->isApplicationStarted.load() ? "started" : "stopped") }
    };
    const std::string application_status_response = application_status_json.dump();
    this->broadcastNotification(application_status_response);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

void Server::sendRamNotification()
{
  while (this->isApplicationStarted) {
    const RAMStats &ram_stats = this->hardwareMonitor.getRamStats();

    const nlohmann::json ram_notification = {
      { "event", "ram stats" },
      { "total", ram_stats.total },
      { "used", ram_stats.used }
    };
    
    const std::string ram_notification_str = ram_notification.dump();
    this->broadcastNotification(ram_notification_str);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void Server::handleNewFrame(cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections)
{
  const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_since_person_detection = now - this->lastPersonNotificationTime;
  const double time_since_person_detection_seconds = time_since_person_detection.count();

  if (time_since_person_detection_seconds >= 15) {
    for (const HailoDetectionPtr &detection : detections) {
      if (detection->get_label() == "person" && detection->get_confidence() > 0.5f) {
        const nlohmann::json person_notification = {{ "event", "person detected"}};
        const std::string person_notification_str = person_notification.dump();
        this->broadcastNotification(person_notification_str);
        this->lastPersonNotificationTime = now;
        break; // Only send one notification per frame
      }
    }
  }

  const std::optional<Buffer> jpeg_buffer = this->convertFrameToJpeg(frame);

  if (jpeg_buffer) {
    this->loop->defer(
      [this, jpeg_buffer]()
      {
        std::string_view string_view = std::string_view((char *)(*jpeg_buffer).data(), (*jpeg_buffer).size());
        this->app.publish("image-stream", string_view, uWS::BINARY);
      }
    );
  }
}

template <bool SSL>
void Server::logRequest(uWS::HttpResponse<SSL> *res, uWS::HttpRequest *req) const
{
  std::lock_guard<std::mutex> lock(io_mutex);
  std::cout << BOLDMAGENTA << "\nIncoming request: " << req->getMethod() << " " << req->getUrl();
  std::cout << std::endl << RESET;
}

void Server::setupRoutes()
{
  this->app
    .post(
      "/start",
      [this](auto *res, auto *req)
      {
        this->threadPool.enqueue(
          [this]()
          {
            if (!this->isApplicationStarted.exchange(true)) {
              this->hailoDeviceTemperatureThread = std::thread(&Server::sendHailoDeviceTemperatureNotification, this);
              this->cpuThread = std::thread(&Server::sendCpuStatsNotification, this);
              this->ramThread = std::thread(&Server::sendRamNotification, this);
              this->camera.startFrameCapture();
              this->detectionInference.start();

              std::lock_guard<std::mutex> lock(this->startStopMutex);
              this->lastStartTime = std::chrono::high_resolution_clock::now();
            }
          }
        );

        nlohmann::json responseJson = {{ "message", "application started"}};
        std::string response = responseJson.dump();
        res->writeStatus("200 OK")->writeHeader("Content-Type", "application/json")->end(response);
        this->logRequest(res, req);
      }
    )
    .post(
      "/stop",
      [this](auto *res, auto *req)
      {
        this->threadPool.enqueue(
          [this, res, req]()
          {
            std::lock_guard<std::mutex> lock(this->startStopMutex);
            const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed_time = now - this->lastStartTime;
            const double elapsed_time_seconds = elapsed_time.count();

            if (elapsed_time_seconds < 10) {
              {
                std::lock_guard<std::mutex> lock(io_mutex);
                std::cerr << BOLDRED << "\nError: Cannot stop detection inference within 10 seconds of starting\n";
                std::cerr << "Elapsed time: " << elapsed_time_seconds << " seconds" << std::endl << RESET;
              }

              nlohmann::json response_json = {{ "message", "Cannot stop detection inference within 10 seconds of starting" }};
              std::string response = response_json.dump();
              res->writeStatus("503 Service Unavailable")->writeHeader("Content-Type", "application/json")->end(response);
              this->logRequest(res, req);

              return;
            }

            if (this->isApplicationStarted.exchange(false)) {
              this->detectionInference.stop();
              this->camera.stopFrameCapture();

              if (this->hailoDeviceTemperatureThread.joinable()) {
                this->hailoDeviceTemperatureThread.join();
              }
              
              if (this->cpuThread.joinable()) {
                this->cpuThread.join();
              }

              if (this->ramThread.joinable()) {
                this->ramThread.join();
              }
            }
          }
        );

        nlohmann::json response_json = {{ "message", "application stopped" }};
        std::string response = response_json.dump();
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
            {
              std::lock_guard<std::mutex> lock(this->clientsMutex);
              this->clients.insert(ws);
            }
            {
              std::lock_guard<std::mutex> lock(io_mutex);
              std::cout << BOLDMAGENTA << "\nClient connected. Thread: " << std::this_thread::get_id();
              std::cout << std::endl << RESET;
            }
            ws->subscribe("image-stream");
          },
        .message =
          [this](auto *ws, std::string_view message, uWS::OpCode op_code)
          {
            {
              std::lock_guard<std::mutex> lock(io_mutex);
              std::cout << BOLDMAGENTA << "\nMessage received: " << message << std::endl << RESET;
            }
            this->loop->defer([ws, op_code] { ws->send("ACK", op_code); });
          },
        .close =
          [this](auto *ws, int code, std::string_view message)
          {
            {
              std::lock_guard<std::mutex> lock(this->clientsMutex);
              this->clients.erase(ws);
            }
            {
              std::lock_guard<std::mutex> lock(io_mutex);
              std::cout << BOLDMAGENTA << "\nClient disconnected. Thread: " << std::this_thread::get_id();
              std::cout << std::endl << RESET;
            }
          }
      }
    );
}

Server& Server::getInstance(const uint8_t jpeg_frame_quality = 50)
{
  static Server instance(jpeg_frame_quality);
  return instance;
}

void Server::startServer(int port = 9898)
{
  this->app
    .listen(
      port,
      [this](auto *listen_socket)
      {
        if (listen_socket) {
          // us_socket_local_port retrieves the port number from the listen socket
          {
            std::lock_guard<std::mutex> lock(io_mutex);
            std::cout << BOLDMAGENTA << "\nThread " << std::this_thread::get_id() << " listening on port ";
            std::cout << us_socket_local_port(true, (struct us_socket_t *)listen_socket) << std::endl;
            std::cout << RESET;
          }
          this->isServerStarted = true;
          this->applicationStatusThread = std::thread(&Server::sendApplicationStatus, this);
        }
        else {
          std::lock_guard<std::mutex> lock(io_mutex);
          std::cout << BOLDRED << "\nThread: " << std::this_thread::get_id() << " failed to listen on port 9898";
          std::cout << std::endl << RESET;
        }
      }
    )
    .run();

    this->isServerStarted = false;

    if (this->applicationStatusThread.joinable()) {
      this->applicationStatusThread.join();
    }

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDMAGENTA << "\nThread: " << std::this_thread::get_id() << " server ended" << std::endl;
    std::cout << RESET;
  }
}

Server::~Server() {
  this->isApplicationStarted.store(false);
  this->isServerStarted.store(false);

  this->threadPool.enqueue(
    [this] {
      this->detectionInference.stop();
      this->camera.stopFrameCapture();
    }
  );

  if (this->applicationStatusThread.joinable()) {
    this->applicationStatusThread.join();
  }

  if (this->hailoDeviceTemperatureThread.joinable()) {
    this->hailoDeviceTemperatureThread.join();
  }
  
  if (this->cpuThread.joinable()) {
    this->cpuThread.join();
  }

  if (this->ramThread.joinable()) {
    this->ramThread.join();
  }

  // Close all client connections
  {
    std::lock_guard<std::mutex> lock(this->clientsMutex);
    for (auto *ws : this->clients) {
      ws->close();
    }
    this->clients.clear();
  }

  // Close all uWebSockets sockets
  this->app.close();

  std::lock_guard<std::mutex> lock(io_mutex);
  std::cout << BOLDMAGENTA << "\nServer cleanup complete" << std::endl << RESET;
}