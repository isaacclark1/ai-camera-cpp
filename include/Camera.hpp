// https :  // github.com/erasta/libcamera-opencv/blob/main/main.cpp

/**
 * This file demostrates how to use libcamera with OpenCV to capture and process camera frames.
 * It sets up a camera pipeline, allocates buffers, and processes frames.
 */

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <libcamera/libcamera.h> // libcamera API for camera access

#include <chrono>                     // For measuring time intervals
#include <cstdlib>                    // For general utilities like EXIT_SUCCESS/EXIT_FAILURE
#include <iomanip>                    // For formatted I/O
#include <functional>                 // For std::function, callbacks
#include <iostream>                   // For standard I/O
#include <memory>                     // For smart pointers
#include <opencv4/opencv2/opencv.hpp> // For image processing
#include <thread>                     // For multi-threading support
#include <cstdint>                    // Fixed-width integer types

#include "MappedBuffer.hpp"           // Interface for CPU access to memory-mapped buffers
#include "ThreadPool.hpp"             // Group of worker threads to execute tasks concurrently
#include "DetectionInference.hpp"     // Logic to perform detection inference on image frames

#define TIMEOUT_SEC 3

// Used to store image data
using Buffer = std::vector<unsigned char>;
using FrameReadyCallback = std::function<void(const Buffer &, const std::vector<HailoDetectionPtr> &detections)>;
using CVMiddlewareCallback = std::function<void(cv::Mat &)>;

/**
 * Camera encapsulates the functionality to capture frames from a camera, process them,
 * and provide callbacks.
 */
class Camera
{
  private:
    ThreadPool threadPool;
    DetectionInference &detectionInference;
    FrameReadyCallback frameReadyCallback;
    CVMiddlewareCallback cvMiddlewareCallback;

    // Camera configuration
    uint16_t height = 1080;
    uint16_t width = 1920;
    uint8_t quality = 50;

    std::atomic<bool> isCameraRolling{false};
    std::atomic<bool> isPipelineStarted{false};

    // A callback that processes an OpenCV Mat image and detection results
    void matCallback(cv::Mat mat, std::vector<HailoDetectionPtr> &detections);

  public:
    // Process an individual request from the camera
    void processRequest(libcamera::Request *request);

    // Called when a camera request completes
    void requestComplete(libcamera::Request *request);

    // Returns a human-friendly camera name based on its properties
    std::string cameraName(libcamera::Camera *camera);

    // Converts an OpenCV Mat (frame) to a JPEG image stored in a buffer
    void convertToJpeg(const cv::Mat &frame, std::vector<unsigned char> &buffer, const int quality);

    // Called when a buffer is completely filled with a frame
    void bufferComplete(libcamera::Request *request, libcamera::FrameBuffer *frameBuffer);

    void setFrameReadyCallback(FrameReadyCallback callback)
    {
      frameReadyCallback = std::move(callback);
    }

    void setCVMiddlewareCallback(CVMiddlewareCallback callback)
    {
      cvMiddlewareCallback = std::move(callback);
    }

    int startCameraPipeline();
    int startFrameCapture();
    
    // Finish and clean up the camera pipeline
    int finish();

    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<std::thread> thread;                       // A seperate thread for processing
    libcamera::Stream *cameraStream;
    libcamera::FrameBufferAllocator *frameBufferAllocator;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::vector<std::unique_ptr<libcamera::Request>> requests; // A vector of requests (one per frame)

    Camera()
      : threadPool(4),
        detectionInference(DetectionInference::getInstance(
          "/home/isaac/projects/ai-camera-cpp/network-files/yolov8s.hef", 1080, 1920
        ))
    {
      detectionInference.setMatCallback(
        [this](cv::Mat mat, std::vector<HailoDetectionPtr> &detections)
        {
          this->matCallback(mat, detections);
        }
      );
    }

    ~Camera() {}
};

/**
 * Convert an OpenCV Mat to a JPEG image in memory
 * 
 * This function sets JPEG compression parameters, encodes the image, and stores the
 * result in a buffer
 */
void Camera::convertToJpeg(const cv::Mat &frame, std::vector<unsigned char> &buffer, const int quality)
{
  std::vector<int> compressionParams;
  compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
  compressionParams.push_back(quality);
  compressionParams.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
  compressionParams.push_back(1);
  
  // Encode the frame into JPEG format
  if (!cv::imencode(".jpg", frame, buffer, compressionParams)) {
    std::cout << "Error in Camera::convertToJpeg:" << std::endl;
    std::cout << "Could not encode the image to JPEG format." << std::endl;
    return;
  }
}

/**
 * Process a frame (cv::Mat) and detections.
 * 
 * This function is called by the inference engine when a new frame is processed.
 * It enqueues a task in the thread pool to convert the frame to JPEG and call the frameReadyCallback.
 */
void Camera::matCallback(cv::Mat mat, std::vector<HailoDetectionPtr> &detections)
{
  threadPool.enqueue([mat, detections, this]()
  {
    Buffer jpegBuffer;
    convertToJpeg(mat, jpegBuffer, quality);

    if (frameReadyCallback) {
      frameReadyCallback(jpegBuffer, detections);
    }
  });
}

/**
 * Handle buffer completion.
 * 
 * Called when a frame buffer is completely filled with image data.
 * It maps the buffer into CPU-accessible memory, convert the raw image to a BGR image,
 * flips it (if necessary), and enqueues it for detection inference.
 */
void Camera::bufferComplete(libcamera::Request *request, libcamera::FrameBuffer *frameBuffer)
{
  // Map the frame buffer for reading
  libcamera::MappedFrameBuffer mappedBuffer(frameBuffer, libcamera::MappedFrameBuffer::MapFlag::Read);
  const std::vector<libcamera::Span<uint8_t>> &mem = mappedBuffer.planes();

  // Create an OpenCV Mat from the raw buffer data
  cv::Mat image(height, width, CV_8UC4, (uint8_t *)(mem[0].data()));

  // Convert from BGRA to BGR
  cv::Mat bgrImage;
  cv::cvtColor(image, bgrImage, cv::COLOR_BGRA2BGR);

  // Flip the image upside down
  // cv::flip(bgrImage, bgrImage, -1);

  this->threadPool.enqueue([bgrImage, this]() { this->detectionInference.writeFrameToDevice(bgrImage); });
}

/**
 * Handle request completion.
 * 
 * Called when a camera Request (which holds frame buffers) is completed.
 * It re-queues the request for continued capture and, if needed, processes the frame.
 */
void Camera::requestComplete(libcamera::Request *request)
{
  // If the request was cancelled or camera is not running, ignore.
  if (request->status() == libcamera::Request::RequestCancelled || !isCameraRolling) {
    return;
  }

  // Additional processing or logging could be done here

  // Reuse the request buffers so the same Request can be used again.
  request->reuse(libcamera::Request::ReuseBuffers);

  // Queue the request back to the camera for the next frame
  camera->queueRequest(request);
}

/**
 * Generate a human-readable camera name.
 * 
 * Uses camera properties (like location) to return a friendly name,
 * appending the unique camera ID.
 */
std::string Camera::cameraName(libcamera::Camera *camera)
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

/**
 * Start the camera pipeline.
 * 
 * Initialises the CameraManager, acquires a camera, configures it, allocates
 * frame buffers, sets up requests, and connects signal callbacks.
 */
int Camera::startCameraPipeline()
{
  if (isPipelineStarted) {
    return EXIT_SUCCESS;
  }

  isPipelineStarted = true;

  // Create and start the camera manager (there can only be one per process)
  cameraManager = std::make_unique<libcamera::CameraManager>();
  cameraManager->start();

  // List available cameras
  for (auto const &camera : cameraManager->cameras()) {
    std::cout << " - " << cameraName(camera.get()) << std::endl;
  }

  // Ensure that there is at least one camera available
  if (cameraManager->cameras().empty()) {
    std::cout << "Error: AICamera::startCameraPipeline:\n";
    std::cout << "No cameras were identified on the system." << std::endl;
    cameraManager->stop();
    return EXIT_FAILURE;
  }

  // Acquire the first available camera
  std::string cameraId = cameraManager->cameras()[0]->id();
  camera = cameraManager->get(cameraId);

  if (!camera) {
    std::cerr << "Error: AICamera::startCameraPipeline():\n";
    std::cerr << "Failed to acquire camera" << std::endl;
    return EXIT_FAILURE;
  }

  camera->acquire();

  // Generate a configuration for the viewfinder stream
  std::unique_ptr<libcamera::CameraConfiguration> config = 
    camera->generateConfiguration({ libcamera::StreamRole::Viewfinder });
  libcamera::StreamConfiguration &streamConfig = config->at(0);
  std::cout << "Default viewfinder configuration is: " << streamConfig.toString() << std::endl;

  // Adjust the configuration to the desired resolution and format
  streamConfig.size.width = width;
  streamConfig.size.height = height;
  streamConfig.pixelFormat = libcamera::formats::XRGB8888;
  streamConfig.bufferCount = 1;

  // Validate the configuration to ensure all settings are acceptable
  config->validate();

  // Apply the configuration to the camera
  int retConfig = camera->configure(config.get());
  
  if (retConfig) {
    std::cout << "Error: AICamera::startCameraPipeline:\n";
    std::cout << "CAMERA CONFIGURATION FAILED" << std::endl;
    return EXIT_FAILURE;
  }

  // Validate again and print out the final configuration
  config->validate();
  std::cout << "Validated viewfinder configuration is: " << streamConfig.toString() << std::endl;
  camera->configure(config.get());

  // Print out camera controls and properties for debugging
  auto controls = camera->controls();
  auto properties = camera->properties();
  std::cout << "Controls:\n";
  for (auto &control : controls) {
    std::cout << control.first->name() << control.second.toString() << " = " <<
      control.second.def().toString() << "\n";
  }
  std::cout << "Properties:\n";
  for (auto &property : properties) {
    std::cout << property.first << ": " << property.second.toString() << std::endl;
  }

  // Allocate frame buffers for the configured stream
  frameBufferAllocator = new libcamera::FrameBufferAllocator(camera);
  for (libcamera::StreamConfiguration &cfg : *config) {
    int ret = frameBufferAllocator->allocate(cfg.stream());

    if (ret < 0) {
      std::cerr << "Error: AICamera::startCameraPipeline:\n";
      std::cerr << "Can't allocate buffers" << std::endl;
      return EXIT_FAILURE;
    }

    size_t allocated = frameBufferAllocator->buffers(cfg.stream()).size();
    std::cout << "Allocated " << allocated << " buffers for stream" << std::endl;
  }

  // Create capture requests for each buffer adn configure them.
  // A request is used to submit buffers to the camera for image capture.
  cameraStream = streamConfig.stream();
  const std::vector<std::unique_ptr<libcamera::FrameBuffer>> &buffers
    = frameBufferAllocator->buffers(cameraStream);
  
  for (unsigned int i = 0; i < buffers.size(); ++i) {
    std::unique_ptr<libcamera::Request> request = camera->createRequest();

    if (!request) {
      std::cerr << "AICamera::startCameraPipeline()\n";
      std::cerr << "Can't create request" << std::endl;
      return EXIT_FAILURE;
    }

    const std::unique_ptr<libcamera::FrameBuffer> &buffer = buffers[i];

    for (auto &plane : buffer->planes()) {
      std::cout << "Buffer " << i << " length " << plane.length << " at " << plane.offset << std::endl;
    }

    int ret = request->addBuffer(cameraStream, buffer.get());

    if (ret < 0) {
      std::cerr << "Error: AICamera::startCameraPipeline\n";
      std::cerr << "Can't set buffer for request" << std::endl;
      return EXIT_FAILURE;
    }

    // Set camera controls for this request
    request->controls().set(libcamera::controls::AeEnable, true);     // Enable auto exposure
    request->controls().set(libcamera::controls::AwbEnable, true);    // Enable auto white balance
    request->controls().set(libcamera::controls::Brightness, 0.0);    // No brightness offset
    request->controls().set(libcamera::controls::Contrast, 1.0);      // Default contrast
    request->controls().set(libcamera::controls::ExposureTime, 0);    // 0 means auto exposure.
    request->controls().set(libcamera::controls::AnalogueGain, 1.0);  // No amplification in sensor signal

    // Store the request for future reuse
    requests.push_back(std::move(request));

    threadPool.enqueue([this]() { this->detectionInference.start(); });
  }

  // Connect the camera's signals to the callback functions to handle completed requests and buffers
  camera->requestCompleted.connect(this, &Camera::requestComplete);
  camera->bufferCompleted.connect(this, &Camera::bufferComplete);

  // Start capturing frames from the camera
  camera->start();
  std::cout << "Camera started" << std::endl;

  return EXIT_SUCCESS;
}

/**
 * Queue all pre-prepared requests to the camera and run the capture loop
 */
int Camera::startFrameCapture()
{
  if (isCameraRolling) {
    return EXIT_SUCCESS;
  }

  isCameraRolling = true;

  // Queue each request to the camera for capture
  for (std::unique_ptr<libcamera::Request> &request : requests) {
    std::cout << "Queued " << (void *)request.get() << std::endl;
    camera->queueRequest(request.get());
    std::cout << "Request queued to camera" << std::endl;
  }

  // !-- An event loop could be run here to process device events
  return EXIT_SUCCESS;
}

/**
 * Clean up and shut down.
 * 
 * Stops the camera, releases resources, disconnects signals, and cleans up memory.
 */
int Camera::finish()
{
  if (!isCameraRolling) {
    return EXIT_SUCCESS;
  }

  isCameraRolling = false;
  isPipelineStarted = false;

  // Stop the camera capture pipeline
  camera->stop();

  // Stop any ongoing inference processing
  this->detectionInference.stop();

  // Disconnect the signal connections
  camera->requestCompleted.disconnect(this, &Camera::requestComplete);
  camera->bufferCompleted.disconnect(this, &Camera::bufferComplete);

  // Free allocated buffers and clean up the allocator
  frameBufferAllocator->free(cameraStream);
  delete frameBufferAllocator;
  frameBufferAllocator = nullptr;
  
  // Release the camera and reset pointers
  camera->release();
  camera.reset();

  // Clear any pending requests
  requests.clear();

  // Reset the stream pointer
  cameraStream = nullptr;

  // Stop and reset the CameraManager
  cameraManager->stop();
  cameraManager.reset();

  std::cout << "AICamera::finish: Clean-up complete" << std::endl;

  return EXIT_SUCCESS;
}

#endif // CAMERA_HPP