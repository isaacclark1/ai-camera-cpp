// https :  // github.com/erasta/libcamera-opencv/blob/main/main.cpp

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <libcamera/libcamera.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <functional>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>
#include <cstdint>
#include <stdexcept>

#include "MappedBuffer.hpp"
#include "ThreadPool.hpp"
#include "DetectionInference.hpp"
#include "global/io_mutex.hpp"
#include "global/log_colours.hpp"

#define TIMEOUT_SEC 3

/**
 * Camera encapsulates the functionality to capture frames from a camera, process them,
 * and provide callbacks.
 */
class Camera
{
  private:
    // Camera configuration
    const uint16_t frameHeight;
    const uint16_t frameWidth;

    uint64_t frameCount{0};
    const uint8_t frameSkipFactor;

    ThreadPool threadPool;
    DetectionInference &detectionInference;

    std::shared_ptr<libcamera::Camera> camera;
    libcamera::Stream *cameraStream;
    libcamera::FrameBufferAllocator *frameBufferAllocator;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::vector<std::unique_ptr<libcamera::Request>> requests; // A vector of requests (one per frame)

    std::atomic<bool> isCameraStarted{false};

    /**
    * Set up the camera.
    * 
    * Initialises the CameraManager, acquires a camera, configures it, allocates
    * frame buffers, sets up requests, and connects signal callbacks.
    * 
    * @param frame_height The height of the frames to store from the camera.
    * @param frame_width The width of the frames to store from the camera.
    * @param frame_skip_factor The factor to skip detection inference frames by (for performance reasons).
    */
    Camera(
      uint16_t frame_height,
      uint16_t frame_width,
      uint8_t frame_skip_factor
    );

    /**
     * Requeue the capture request to the camera for continued capture.
     * It is called when a capture request is completed.
     * 
     * @param request The request that was completed.
     */
    void requestCompleteCallback(libcamera::Request *request);

    /**
     * Print a human friendly camera name.
     * 
     * @param camera The camera to get the name for.
     * @return A human friendly camera name.
     */
    std::string printCameraName(libcamera::Camera *camera);

    /**
     * Called when a buffer is completely filled with image data.
     * An image is created from the buffer and written to the Hailo device for inference.
     * 
     * @param request The request that the buffer belongs to.
     * @param frameBuffer The frame buffer that is filled.
     */
    void bufferCompleteCallback(libcamera::Request *request, libcamera::FrameBuffer *frameBuffer);

    /**
     * Print a list of available cameras
     */
    void logCameras();

    /**
     * Acquire the first available camera
     */
    void acquireCamera();

    /**
     * Configure the camera 
     */
    void configureCamera();

    /**
     * Print camera controls and properties.
     */
    void logCameraProperties();

    /**
     * Create camera capture requests for the camera stream and configure them.
     */
    void createCaptureRequests();

    /**
     * Allocate frame buffers for the camera stream.
     * 
     * @param config The camera configuration.
     */
    void allocateFrameBuffers(std::unique_ptr<libcamera::CameraConfiguration> config);

  public:
    // Delete copy constructor and assignment operator to prevent copying
    Camera(const Camera &) = delete;
    Camera& operator=(const Camera &) = delete;

    /**
    * Static method to get the singleton instance.
    * If the instance hasn't been created, it will be created and returned.
    * 
    * @param frame_height The height of the frames to store from the camera.
    * @param frame_width The width of the frames to store from the camera.
    * @param frame_skip_factor The factor to skip detection inference frames by (for performance reasons).
    * @return The singleton instance of the Camera class.
    */
    static Camera& getInstance(
      uint16_t frame_height,
      uint16_t frame_width,
      uint8_t frame_skip_factor
    );

    /**
     * Queue all pre-prepared requests to the camera and run the capture loop
     * 
     * @return EXIT_SUCCESS if the camera is started, EXIT_FAILURE otherwise.
     */
    int startFrameCapture();

    /**
     * Stop frame capture.
     */
    int stopFrameCapture();

    /**
     * Destroy the Camera gracefully.
     */
    ~Camera();
};

Camera::Camera(
  uint16_t frame_height,
  uint16_t frame_width,
  uint8_t frame_skip_factor
)
  : frameHeight(frame_height),
    frameWidth(frame_width),
    frameSkipFactor(frame_skip_factor),
    threadPool(4),
    cameraManager(std::make_unique<libcamera::CameraManager>()),
    detectionInference(DetectionInference::getInstance())
{
  this->cameraManager->start();

  this->logCameras();
  this->acquireCamera();
  this->configureCamera();

  std::cout << BOLDGREEN <<  "\nCamera setup complete\n" << std::endl << RESET;
}

Camera& Camera::getInstance(
  uint16_t frame_height = 1080,
  uint16_t frame_width = 1920,
  uint8_t frame_skip_factor = 1
)
{
  static std::mutex instanceMutex;
  std::lock_guard<std::mutex> lock(instanceMutex);

  static Camera instance(
    frame_height, frame_width, frame_skip_factor
  );
  return instance;
}

Camera::~Camera()
{
  this->stopFrameCapture();

  // Free allocated buffers
  this->frameBufferAllocator->free(this->cameraStream);
  delete this->frameBufferAllocator;
  this->frameBufferAllocator = nullptr;

  this->camera->release();
  this->camera.reset();

  this->requests.clear();
  delete this->cameraStream;
  this->cameraStream = nullptr;

  this->cameraManager->stop();
  cameraManager.reset();

  std::cout << BOLDGREEN << "\nCamera cleanup complete\n" << std::endl << RESET;
}

void Camera::bufferCompleteCallback(libcamera::Request *request, libcamera::FrameBuffer *frameBuffer)
{
  // Map the frame buffer for reading
  libcamera::MappedFrameBuffer mappedBuffer(frameBuffer, libcamera::MappedFrameBuffer::MapFlag::Read);
  const std::vector<libcamera::Span<uint8_t>> &mem = mappedBuffer.planes();

  // Create an OpenCV Mat from the raw buffer data
  cv::Mat image(this->frameHeight, this->frameWidth, CV_8UC4, (uint8_t *)(mem[0].data()));

  // Convert from BGRA to BGR
  cv::Mat bgrImage;
  cv::cvtColor(image, bgrImage, cv::COLOR_BGRA2BGR);

  // Flip the image upside down
  // cv::flip(bgrImage, bgrImage, -1);
  if (this->frameCount % this->frameSkipFactor == 0) {
    this->threadPool.enqueue([bgrImage, this]() { this->detectionInference.writeFrameToHailoDevice(bgrImage); });
  }
  ++this->frameCount;
}

void Camera::requestCompleteCallback(libcamera::Request *request)
{
  // If the request was cancelled or camera is not running, ignore.
  if (request->status() == libcamera::Request::RequestCancelled || !this->isCameraStarted) {
    return;
  }

  // Additional processing or logging could be done here

  // Reuse the request buffers so the same Request can be used again.
  request->reuse(libcamera::Request::ReuseBuffers);

  // Queue the request back to the camera to get the next frame
  this->camera->queueRequest(request);
}

std::string Camera::printCameraName(libcamera::Camera *camera)
{
  const libcamera::ControlList &props = camera->properties();
  std::string name;
  // Get the camera location property
  std::optional<uint32_t> locationOption = props.get(libcamera::properties::Location);
  
  // Determine the camera location and assign a corresponding name
  if (locationOption) {
    switch (*locationOption) {
      case libcamera::properties::CameraLocationFront:
        name = "Internal front camera";
        break;
      case libcamera::properties::CameraLocationBack:
        name = "Internal back camera";
        break;
      case libcamera::properties::CameraLocationExternal:
        name = "Extenal camera";
        break;
      default:
        name = "Unknown camera location";
        break;
    }
  }
  else {
    name = "Unknown camera location";
  }

  name += " (" + camera->id() + ")";
  return name;
}

int Camera::startFrameCapture()
{
  if (this->isCameraStarted) {
    return EXIT_SUCCESS;
  }

  // Connect the camera's signals to the callback functions to handle completed requests and buffers
  this->camera->requestCompleted.connect(this, &Camera::requestCompleteCallback);
  this->camera->bufferCompleted.connect(this, &Camera::bufferCompleteCallback);

  // Start capturing frames from the camera
  this->camera->start();

  this->isCameraStarted = true;

  this->createCaptureRequests();

  // Queue each request to the camera for capture
  for (std::unique_ptr<libcamera::Request> &request : this->requests) {
    camera->queueRequest(request.get());
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDGREEN << "Request: " << (void *)request.get() << " queued to camera" << std::endl << RESET;
  }

  std::cout << BOLDGREEN << "\nCamera frame capture started" << std::endl << RESET;

  // !-- An event loop could be run here to process device events
  return EXIT_SUCCESS;
}

int Camera::stopFrameCapture()
{
  if (!this->isCameraStarted) {
    return EXIT_SUCCESS;
  }

  // Disconnect callbacks - to stop capturing frames before stopping the camera!
  this->camera->requestCompleted.disconnect(this, &Camera::requestCompleteCallback);
  this->camera->bufferCompleted.disconnect(this, &Camera::bufferCompleteCallback);

  this->camera->stop();

  this->isCameraStarted = false;

  this->requests.clear();

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDGREEN << "\nCamera frame capture stopped\n" << std::endl << RESET;
  }

  return EXIT_SUCCESS;
}

void Camera::logCameras()
{
  for (auto const &camera : this->cameraManager->cameras()) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDGREEN << "\nCamera location: " << this->printCameraName(camera.get()) << std::endl << RESET;
  }
}

void Camera::acquireCamera()
{
  // Ensure that there is at least one camera available
  if (this->cameraManager->cameras().empty()) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDRED << "\nError: Camera::acquireCamera:\n";
      std::cout << "No cameras were identified on the system." << std::endl << RESET;
    }
    cameraManager->stop();
    throw std::runtime_error("No cameras were identified on the system.");
  }

  // Acquire the first available camera
  std::string cameraId = this->cameraManager->cameras()[0]->id();
  this->camera = this->cameraManager->get(cameraId);

  if (!camera) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cerr << BOLDRED << "\nError: Camera::acquireCamera:\n";
    std::cerr << "Failed to acquire camera" << std::endl << RESET;
    throw std::runtime_error("Failed to acquire camera");
  }

  this->camera->acquire();
}

void Camera::configureCamera()
{
  // Generate a configuration for the viewfinder stream
  auto config = this->camera->generateConfiguration({ libcamera::StreamRole::Viewfinder });
  libcamera::StreamConfiguration &streamConfig = config->at(0);

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDGREEN << "\nDefault viewfinder configuration is: ";
    std::cout  << streamConfig.toString() << std::endl << RESET;
  }

  // Adjust the configuration to the desired resolution and format
  streamConfig.size.width = this->frameWidth;
  streamConfig.size.height = this->frameHeight;
  streamConfig.pixelFormat = libcamera::formats::XRGB8888;
  streamConfig.bufferCount = 1;

  config->validate();

  int retConfig = this->camera->configure(config.get());

  if (retConfig) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDRED << "\nError: Camera::configureCamera:\n";
    std::cout << "Camera configuration failed" << std::endl << RESET;
    throw std::runtime_error("Camera configuration failed");
  }

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDGREEN << "\nValidated viewfinder configuration is: " 
      << streamConfig.toString() << std::endl << RESET;
  }

  this->cameraStream = streamConfig.stream();

  this->logCameraProperties();
  this->allocateFrameBuffers(std::move(config));
}

void Camera::logCameraProperties()
{
  auto controls = this->camera->controls();
  auto properties = this->camera->properties();

  std::cout << BOLDGREEN << "\nCamera controls:\n\n";

  for (auto const &control : controls) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << control.first->name() << control.second.toString() << " = " <<
      control.second.def().toString() << "\n";
  }

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << "\nCamera properties:\n\n";
  }

  for (auto const &property : properties) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << property.first << ": " << property.second.toString() << std::endl;
  }

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << RESET;
  }
}

void Camera::createCaptureRequests()
{
  const auto &buffers = this->frameBufferAllocator->buffers(this->cameraStream);

  for (unsigned int i = 0; i < buffers.size(); ++i) {
    std::unique_ptr<libcamera::Request> request = this->camera->createRequest();

    if (!request) {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "\nCamera::startCameraPipeline()\n";
      std::cerr << "Can't create camera capture request" << std::endl << RESET;
      throw std::runtime_error("Can't create camera capture request");
    }

    const std::unique_ptr<libcamera::FrameBuffer> &buffer = buffers[i];

    for (auto &plane : buffer->planes()) {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDGREEN << "\nBuffer " << i << ": length " << plane.length << " at offset: " << plane.offset << std::endl << RESET;
    }

    int ret = request->addBuffer(cameraStream, buffer.get());

    if (ret < 0) {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "\nError: Camera::createCaptureRequests\n";
      std::cerr << "Can't set buffer for camera capture request" << std::endl << RESET;
      throw std::runtime_error("Can't set buffer for camera capture request");
    }

    // Set camera controls for this request
    request->controls().set(libcamera::controls::AeEnable, true);     // Enable auto exposure
    request->controls().set(libcamera::controls::AwbEnable, true);    // Enable auto white balance
    request->controls().set(libcamera::controls::Brightness, 0.0f);    // No brightness offset
    request->controls().set(libcamera::controls::Contrast, 1.0f);      // Default contrast
    request->controls().set(libcamera::controls::ExposureTime, 0);    // 0 means auto exposure.
    request->controls().set(libcamera::controls::AnalogueGain, 1.0f);  // No amplification in sensor signal

    // Store the request for future reuse
    this->requests.push_back(std::move(request));
  }

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDGREEN << "Successfully created frame capture requests" << std::endl << RESET;
  }
}

void Camera::allocateFrameBuffers(std::unique_ptr<libcamera::CameraConfiguration> config)
{
  // Allocate frame buffers for the configured stream
  this->frameBufferAllocator = new libcamera::FrameBufferAllocator(this->camera);

  for (libcamera::StreamConfiguration &cfg : *config) {
    int ret = this->frameBufferAllocator->allocate(cfg.stream());

    if (ret < 0) {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "\nError: Camera::Camera:\n";
      std::cerr << "Can't allocate buffers for camera stream" << std::endl << RESET;
      throw std::runtime_error("Can't allocate buffers for camera stream");
    }

    size_t allocated = this->frameBufferAllocator->buffers(cfg.stream()).size();

    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDGREEN << "\nAllocated " << allocated << " buffers for camera stream" << std::endl << RESET;
    }
  }
}

#endif // CAMERA_HPP