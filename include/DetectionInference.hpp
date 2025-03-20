#ifndef DETECTION_INFERENCE_HPP
#define DETECTION_INFERENCE_HPP

#include <mutex>
#include <thread>
#include <chrono>
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <common.h>
#include "hailo_objects.hpp"
#include "FrameQueue.hpp"
#include <hailo/hailort.hpp>
#include <regex>
#include "dataset_labels/coco_eighty.hpp"
#include "dataset_labels/coco_ninety.hpp"
#include "hailo_nms_decode.hpp"
#include <stdexcept>
#include "hailo_common.hpp"
#include <unordered_map>
#include "bbox_colours/bbox_colours.hpp"
#include "global/log_colours.hpp"
#include "global/io_mutex.hpp"
#include "hailo_device/HailoDeviceInfo.hpp"
#include <optional>

/**
 * Callback function that processes an OpenCV image (cv::Mat)
 * along with a vector of detection results (HailoDetectionPointer)
 */
using FrameReadyCallback = std::function<void(cv::Mat &, const std::vector<HailoDetectionPtr> &)>;

/**
 * DetectionInference encapsulates the logic to perform detection inference on image frames
 * using the Hailo device.
 */
class DetectionInference
{
  private:
    std::mutex frameQueueMutex;

    const bool QUANTIZED = true; // Reduce precision of numerical values in the neural network
    const hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO; // Automatically determine the format type for the HAILO device
    
    const uint16_t frameHeight;
    const uint16_t frameWidth;

    std::unique_ptr<hailort::VDevice> vdevice; // Hailo VDevice (virtual device) representing the hailo hardware
    std::unique_ptr<std::vector<std::reference_wrapper<hailort::Device>>> hailoDevices; // Phyiscal Hailo devices
    // The loaded neural network that will be used for inference
    std::shared_ptr<hailort::ConfiguredNetworkGroup> networkGroup;
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams;
    std::unique_ptr<FrameQueue<cv::Mat>> frames;
    std::atomic<bool> hasDetectionStarted{false};
    FrameReadyCallback frameReadyCallback;

    std::future<hailo_status> postprocessThread;
    std::vector<std::future<hailo_status>> readFromHailoDeviceThreads;

    std::unordered_map<std::string, cv::Scalar> bboxColours = bbox_colours;
    std::unordered_map<uint8_t, std::string> cocoEightyLabels = common::coco_eighty_classes;
    std::unordered_map<uint8_t, std::string> cocoNinetyLabels = common::coco_ninety_classes;

    HailoDeviceInfo hailoDeviceInfo;

    uint32_t inputVStreamExpectedHeight;
    uint32_t inputVStreamExpectedWidth;

    /**
     * Setup virtual device, network group, and input/output video streams.
     * 
     * @param hef_file The compiled network file for the Hailo device.
     * @param frame_height The height of the frame.
     * @param frame_width The width of the frame.
     */
    DetectionInference(std::string hef_file, uint16_t frame_height, uint16_t frame_width);

    /**
     * Draw detection results on an image frame.
     * 
     * @param frame The image frame to draw on.
     * @param detections The detection results to draw.
     */
    void drawDetections(cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections);

    /**
     * Start and continously postprocess frames using the detection features.
     * 
     * @param features The features to process.
     */
    template <typename T>
    hailo_status continuouslyRunPostprocessing(std::vector<std::shared_ptr<FeatureData<T>>> &features);

    /**
     * Continuously read data from the inference engine.
     * 
     * @param output_vstream The output stream to read from.
     * @param feature The feature data to read into.
     */
    template <typename T>
    hailo_status continuouslyReadInferenceDataFromHailoDevice(
      hailort::OutputVStream &output_vstream, std::shared_ptr<FeatureData<T>> feature
    );

    /**
     * Start the inference process.
     * 
     * @param output_vstream The output stream to read from.
     */
    template <typename T>
    hailo_status startInference(
      std::vector<hailort::OutputVStream> &output_vstream
    );

    /**
     * Print the vstream/network information.
     * 
     * @param vstreams The input and output vstreams.
     */
    void logNetworkBanner(
      hailort::InputVStream &input_vstream, hailort::OutputVStream &output_vstream
    );

    /**
     * Configure the network group for inference.
     * 
     * @param hef_file The inference network compiled for the Hailo device (.hef).
     */
    hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> configureNetworkGroup(
      const std::string hef_file
    );

    /**
     * Filter a region of interest by decoding tensor data using Non-Maximum Suppression (NMS)
     * and then add the resulting detections to the ROI.
     * 
     * @param roi The region of interest to filter.
     */
    void filterROI(HailoROIPtr roi);

    /**
     * Create a feature.
     * 
     * @param vstream_info The video stream information.
     * @param output_frame_size The size of the output frame.
     * @param feature The feature to create.
     */
    template <typename T>
    hailo_status createFeature(
      hailo_vstream_info_t vstream_info,
      size_t output_frame_size,
      std::shared_ptr<FeatureData<T>> &feature
    );

    /**
     * Set the Hailo device information from the first Hailo device found.
     */
    void setHailoDeviceInfo();

    /**
     * Log the Hailo device information.
     */
    void logHailoDeviceInfo();

  public:
    // Delete copy constructor and assignment operator to prevent copying
    DetectionInference(const DetectionInference &) = delete;
    DetectionInference& operator=(const DetectionInference &) = delete;

    /**
     * Static method to get the singleton instance.
     * If the instance hasn't been created, it will be created and returned.
     * 
     * @param hef_file The compiled network file for the Hailo device.
     * @param frame_height The desired height of the frame.
     * @param frame_width The desired width of the frame.
     */
    static DetectionInference& getInstance(std::string hef_file, uint16_t frame_height, uint16_t frame_width);

    /**
     * Gracefully destroy the DetectionInference instance.
     */
    ~DetectionInference();

    /**
     * Start the detection inference
     * 
     * @return A hailo_status indicating the status of the operation.
     */
    hailo_status start();

    /**
     * Stop the detection infererence pipeline.
     * 
     * @return A hailo_status indicating the status of the operation.
     */
    hailo_status stop();

    /**
     * Send an image frame to the inference engine.
     * 
     * @param frame The image frame.
     */
    void writeFrameToHailoDevice(cv::Mat frame);

    /**
      * Set the callback function that is called when an OpenCV Mat and detections are ready.
      * 
      * @param callback The callback function.
      */
    void setFrameReadyCallback(FrameReadyCallback callback);

    /**
     * Get the Hailo device temperature (in celcius).
     */
    std::optional<float32_t> getHailoDeviceTemp();
};

// ===========================================
// 🔹 Constructor / Destructor definitions 🔹
// ===========================================

DetectionInference::DetectionInference(std::string hef_file, uint16_t frame_height, uint16_t frame_width)
  : frames(std::make_unique<FrameQueue<cv::Mat>>(3)),
    frameHeight(frame_height),
    frameWidth(frame_width)
{
  hailort::Expected<std::unique_ptr<hailort::VDevice>> vdevice_exp = hailort::VDevice::create();

  if (!vdevice_exp) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: DetectionInference:\n";
      std::cerr << "Failed to create vdevice, status = " << vdevice_exp.status() << std::endl << RESET;
    }

    throw std::runtime_error("Failed to create vdevice with status: " + std::to_string(vdevice_exp.status()));
  }

  this->vdevice = vdevice_exp.release();

  auto hailo_devices_exp = this->vdevice->get_physical_devices();

  if (!hailo_devices_exp) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: DetectionInference:\n";
      std::cerr << "Failed to get Hailo physical devices, status = " << hailo_devices_exp.status() << std::endl << RESET;
    }

    throw std::runtime_error("Failed to get physical Hailo devices with status: " + std::to_string(hailo_devices_exp.status()));
  }

  this->hailoDevices = std::make_unique<std::vector<std::reference_wrapper<hailort::Device>>>(hailo_devices_exp.release());

  if (this->hailoDevices->empty()) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: DetectionInference:\n";
      std::cerr << "No Hailo devices found" << std::endl << RESET;
    }

    throw std::runtime_error("No Hailo devices found");
  }

  this->setHailoDeviceInfo();
  this->logHailoDeviceInfo();

  auto network_group_exp = this->configureNetworkGroup(hef_file);

  if (!network_group_exp) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: DetetionInference:\n";
      std::cerr << "Failed to configure network group: " << hef_file << std::endl << RESET;
    }

    throw std::runtime_error(
      "Failed to configure network group with status: " + std::to_string(network_group_exp.status())
    );
  }

  this->networkGroup = network_group_exp.release();
  auto vstreams_exp = hailort::VStreamsBuilder::create_vstreams(*networkGroup, QUANTIZED, FORMAT_TYPE);
  
  if (!vstreams_exp) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: DetectionInference:\n";
      std::cerr << "Failed to create vstreams: " << vstreams_exp.status() << std::endl << RESET;
    }

    throw std::runtime_error("Failed to create vstreams with status: " + std::to_string(vstreams_exp.status()));
  }

  this->vstreams = vstreams_exp.release();
  this->logNetworkBanner(this->vstreams.first[0], this->vstreams.second[0]);

  hailo_3d_image_shape_t inputVstreamExpectedShape = this->vstreams.first[0].get_info().shape;
  this->inputVStreamExpectedHeight = inputVstreamExpectedShape.height;
  this->inputVStreamExpectedWidth = inputVstreamExpectedShape.width;

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "DetectionInference setup complete" << std::endl << RESET;
  }
}

DetectionInference::~DetectionInference()
{
    // Clear input and output vstreams
    this->vstreams.first.clear();
    this->vstreams.second.clear();

    this->networkGroup.reset();
    this->vdevice.reset();

    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDBLUE << "\nDetectionInference cleanup complete\n" << std::endl << RESET;
    }
}


// ==================================
// 🔹 Member function Definitions 🔹
// ==================================

DetectionInference& DetectionInference::getInstance(
  std::string hef_file = "",
  uint16_t frame_height = 1080,
  uint16_t frame_width = 1920
)
{
  static std::mutex instanceMutex;
  std::lock_guard<std::mutex> lock(instanceMutex);

  static DetectionInference instance(hef_file, frame_height, frame_width);
  return instance;
}

void DetectionInference::setHailoDeviceInfo()
{
  hailort::Device &device = this->hailoDevices->front().get();

  // Get device type
  const char *id_exp = device.get_dev_id();
  const std::string id = id_exp;
  auto device_type = device.get_type();
  std::string device_type_str = "<UNKNOWN>";
    
  switch (device_type) {
    case hailort::Device::Type::PCIE:
      device_type_str = "PCIe";
      break;
    case hailort::Device::Type::ETH:
      device_type_str = "ETH";
      break;
    case hailort::Device::Type::INTEGRATED:
      device_type_str = "INTEGRATED";
      break;
  }

  // Get device architecture
  auto device_architecture_exp = device.get_architecture();
  hailo_device_architecture_t device_architecture = hailo_device_architecture_t::HAILO_ARCH_MAX_ENUM; // Max enum value as placeholder
  std::string device_architecture_str = "<UNKNOWN>";

  if (!device_architecture_exp) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cerr << BOLDRED << "Error: DetectionInference::getHailoDeviceInfo():\n";
    std::cerr << "Failed to get device architecture for device: " << device_type_str << std::endl << RESET;
  }
  else {
    device_architecture = device_architecture_exp.release();

    switch (device_architecture) {
      case hailo_device_architecture_t::HAILO_ARCH_HAILO8_A0:
        device_architecture_str = "HAILO8_A0";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_HAILO8:
        device_architecture_str = "HAILO8";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_HAILO8L:
        device_architecture_str = "HAILO8L";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_HAILO15H:
        device_architecture_str = "HAILO15H";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_HAILO15L:
        device_architecture_str = "HAILO15L";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_HAILO15M:
        device_architecture_str = "HAILO15M";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_HAILO10H:
        device_architecture_str = "HAILO10H";
        break;
      case hailo_device_architecture_t::HAILO_ARCH_MAX_ENUM:
        device_architecture_str = "<ERROR>";
        break;
    }
  }

  this->hailoDeviceInfo = HailoDeviceInfo(
    id,
    device_type_str,
    device_architecture_str
  );
}

void DetectionInference::logHailoDeviceInfo()
{
  using namespace std;
  lock_guard<mutex> lock(io_mutex);
  cout << BOLDBLUE                                                                               << "\n";
  cout << "====================================================================================" << "\n";
  cout << "     Hailo Device Information:"                                                       << "\n";
  cout << "                                Device ID: "
       << this->hailoDeviceInfo.id                                                               << "\n";
  cout << "                                Device Type: "
       << this->hailoDeviceInfo.type                                                             << "\n";
  cout << "                                Device Architecture: " 
       << this->hailoDeviceInfo.architecture                                                     << "\n";
  cout << "====================================================================================" << "\n";
  cout << endl << RESET;
}

std::optional<float32_t> DetectionInference::getHailoDeviceTemp()
{
  hailort::Device &device = this->hailoDevices->front().get();

  // Get device temperature
  auto device_temp_exp = device.get_chip_temperature();
  hailo_chip_temperature_info_t device_temp;

  if (!device_temp_exp) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cerr << BOLDRED << "Error: DetectionInference::getHailoDeviceStats():\n";
    std::cerr << "Failed to get device temperature" << std::endl << RESET;
    return std::nullopt;
  }
  else {
    device_temp = device_temp_exp.release();
  }
  return device_temp.ts0_temperature;
}

void DetectionInference::drawDetections(cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections)
{
  // Iterate through each detection
  for (const HailoDetectionPtr &detection : detections) {
    // Skip detections with less than 25% confidence
    if (detection->get_confidence() < 0.25f) {
      continue;
    }

    // Get the bounding box (normalised values) of the detection
    HailoBBox bbox = detection->get_bbox();

    // Convert normalised coordinates to absolute pixel coordinates
    float xmin = bbox.xmin() * static_cast<float>(this->frameWidth);
    float ymin = bbox.ymin() * static_cast<float>(this->frameHeight);
    float xmax = bbox.xmax() * static_cast<float>(this->frameWidth);
    float ymax = bbox.ymax() * static_cast<float>(this->frameHeight);

    std::string label = detection->get_label();
    
    cv::Scalar colour = this->bboxColours[label];

    // Draw the bounding box on the frame
    cv::rectangle(frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), colour, 2);

    // Round the most significant digit from the detection confidence to the nearest integer
    uint8_t confidence = static_cast<uint8_t>(detection->get_confidence() * 100);
    
    std::string text = label + " " + std::to_string(confidence) + "%";

    int baseline = 0;
    int textSize = 1;
    int textThickness = 2;
    cv::Size2f text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, textSize, textThickness, &baseline);

    // Define the position for the text (above the bounding box)
    cv::Point2f text_position(xmin, ymin - 10);

    // Draw a filled rectangle as background for the text
    cv::rectangle(
      frame,
      cv::Point2f(text_position.x, text_position.y - text_size.height - static_cast<float>(baseline)),
      cv::Point2f(text_position.x + text_size.width, text_position.y + static_cast<float>(baseline)),
      colour,
      cv::LineTypes::FILLED
    );

    // Put the label text over the background rectangle
    cv::putText(frame, text, text_position, cv::FONT_HERSHEY_SIMPLEX, textSize, cv::Scalar(255, 255, 255), textThickness);
  }
}

template <typename T>
hailo_status DetectionInference::continuouslyRunPostprocessing(std::vector<std::shared_ptr<FeatureData<T>>> &features)
{
  // Sort feature data by tensor size
  std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "\nRunning postprocessing" << std::endl << RESET;
  }

  while (this->hasDetectionStarted) {
    // Create a region of interest (ROI) that covers the whole image
    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

    // Add tensors to the ROI from each feature
    for (uint i = 0; i < features.size(); i++) {
      roi->add_tensor(std::make_shared<HailoTensor>(
        reinterpret_cast<T *>(features[i]->m_buffers.get_read_buffer().data()), features[i]->m_vstream_info
      ));
    }

    // Filter detections using the ROI
    this->filterROI(roi);

    // Release the read buffers for each feature
    for (auto &feature : features) {
      feature->m_buffers.release_read_buffer();
    }

    // Get detection results from the ROI
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    cv::Mat first_frame;

    // Get the frame from the front of the queue
    {
      std::lock_guard<std::mutex> lock(this->frameQueueMutex);

      if (this->frames->isEmpty()) {
        std::cout << BOLDYELLOW << "DetectionInference::continuouslyRunPostprocessing():" << "\n";
        std::cout << "Frame queue is empty. Skipping postprocessing..." << std::endl << RESET;
        continue;
      }

      first_frame = this->frames->front();
    }

    // Resize the frame to the original dimensions
    cv::resize(first_frame, first_frame, cv::Size(this->frameWidth, this->frameHeight));

    // Draw the detection results on the frame
    this->drawDetections(first_frame, detections);

    // Call the callback function if it is set
    if (this->frameReadyCallback != nullptr) {
      this->frameReadyCallback(first_frame, detections);
    }

    // Release the current frame and remove it from the queue
    first_frame.release();

    {
      std::lock_guard<std::mutex> lock(this->frameQueueMutex);
      this->frames->dequeue();
    }
  }

  return HAILO_SUCCESS;
}

template <typename T>
hailo_status DetectionInference::continuouslyReadInferenceDataFromHailoDevice(
  hailort::OutputVStream &output_vstream, std::shared_ptr<FeatureData<T>> feature
)
{
  while (this->hasDetectionStarted) {
    std::vector<T> &buffer = feature->m_buffers.get_write_buffer();
    // Read data from the output stream into the buffer
    hailo_status status = output_vstream.read(hailort::MemoryView(buffer.data(), buffer.size()));
    // Release the write bufer once the read is complete
    feature->m_buffers.release_write_buffer();

    if (HAILO_SUCCESS != status) {
      {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDYELLOW << "Error: detection_inference: continuouslyReadInferenceDataFromHailoDevice():\n";
        std::cerr << "Failed reading with status = " << status << std::endl << RESET;
      }

      return status;
    }
  }

  return HAILO_SUCCESS;
}

void DetectionInference::writeFrameToHailoDevice(cv::Mat frame)
{
  // Resize the input frame to the dimensions required by the network
  cv::resize(frame, frame, cv::Size(this->inputVStreamExpectedWidth, this->inputVStreamExpectedHeight), 1);

  {
    std::lock_guard<std::mutex> lock(this->frameQueueMutex);
    this->frames->enqueue(frame);
  }

  cv::Mat frontFrame;
  bool isFrameQueueEmpty = false;

  {
    std::lock_guard<std::mutex> lock(this->frameQueueMutex);
    isFrameQueueEmpty = this->frames->isEmpty();
    
    if (!isFrameQueueEmpty) {
      frontFrame = this->frames->front();
    }
  }

  if (isFrameQueueEmpty) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDYELLOW << "DetectionInference::writeFrameToHailoDevice:\n";
      std::cout << "Frame queue is empty" << std::endl << RESET;
    }
    return;
  }

  // Verify that the front frame is valid
  if (frontFrame.empty() || frontFrame.data == nullptr) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDYELLOW << "DetectionInference::writeFrameToHailoDevice:\n";
      std::cout << "Invalid frame" << std::endl << RESET;
    }
    return;
  }

  hailort::InputVStream &input_vstream = this->vstreams.first[0];
  
  hailo_status write_status = input_vstream.write(hailort::MemoryView(
    frontFrame.data,
    input_vstream.get_frame_size()
  ));

  if (write_status != HAILO_SUCCESS) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDYELLOW << "\nDetectionInference::writeFrameToHailoDevice:\n";
      std::cout << "Status: " << write_status << " -> Failed to write frame to input stream" << std::endl << RESET;
    }

    return;
  }
}

template <typename T>
hailo_status DetectionInference::createFeature(
  const hailo_vstream_info_t vstream_info,
  const size_t output_frame_size,
  std::shared_ptr<FeatureData<T>> &feature
)
{
  feature = std::make_shared<FeatureData<T>>(
    static_cast<uint32_t>(output_frame_size),
    vstream_info.quant_info.qp_zp,
    vstream_info.quant_info.qp_scale,
    vstream_info.shape.width,
    vstream_info
  );

  return HAILO_SUCCESS;
}

template <typename T>
hailo_status DetectionInference::startInference(
  std::vector<hailort::OutputVStream> &output_vstream
)
{
  std::cout << BOLDBLUE << "Starting inference" << std::endl << RESET;

  hailo_status status = HAILO_UNINITIALIZED;
  size_t output_vstream_size = output_vstream.size();

  // Create feature data for each output stream
  std::vector<std::shared_ptr<FeatureData<T>>> features;
  features.reserve(output_vstream_size);

  for (size_t i = 0; i < output_vstream_size; i++) {
    std::shared_ptr<FeatureData<T>> feature(nullptr);
    hailo_status status = this->createFeature(output_vstream[i].get_info(), output_vstream[i].get_frame_size(), feature);

    if (HAILO_SUCCESS != status) {
      {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDRED << "Error: detection_inference: run_inference\n";
        std::cerr << "Failure creating feature with status = " << status << std::endl << RESET;
      }
      return status;
    }

    features.emplace_back(feature);
  }

  this->readFromHailoDeviceThreads.reserve(output_vstream_size);

  for (size_t i = 0; i < output_vstream_size; i++) {
    this->readFromHailoDeviceThreads.emplace_back(
      std::async(
        std::launch::async,
        &DetectionInference::continuouslyReadInferenceDataFromHailoDevice<T>,
        this,
        std::ref(output_vstream[i]),
        features[i]
      )
    );
  }

  this->postprocessThread = std::async(
    std::launch::async,
    &DetectionInference::continuouslyRunPostprocessing<T>,
    this,
    std::ref(features)
  );

  // Wait for all read threads to complete
  for (size_t i = 0; i < this->readFromHailoDeviceThreads.size(); ++i) {
    status = this->readFromHailoDeviceThreads[i].get();
  }

  hailo_status postprocess_status = this->postprocessThread.get(); // Wait for postprocessing thread to complete
  
  if (HAILO_SUCCESS != status) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference::startInference():\n";
      std::cerr << "Read failed with status: " << status << std::endl << RESET;
    }
    return status;
  }

  if (HAILO_SUCCESS != postprocess_status) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference::startInference():\n";
      std::cerr << "Post-processing failed with status " << postprocess_status << std::endl << RESET;
    }
    return postprocess_status;
  }


  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "Inference finished successfully" << RESET << std::endl;
  }

  return HAILO_SUCCESS;
}

void DetectionInference::logNetworkBanner(
  hailort::InputVStream &input_vstream, hailort::OutputVStream &output_vstream
)
{
  std::lock_guard<std::mutex> lock(io_mutex);

  std::cout << BOLDBLUE << "\n";
  std::cout << "====================================================================================" << "\n";
  std::cout << "     Inference Network Name:"                                                         << "\n";
  std::cout << "                              IN: " << input_vstream.name()                           << "\n";
  std::cout << "                              OUT: " << output_vstream.name()                         << "\n";
  std::cout << "====================================================================================";
  std::cout << std::endl << RESET;
}

hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> DetectionInference::configureNetworkGroup(
  const std::string hef_file
)
{
  hailort::Expected<hailort::Hef> hef_exp = hailort::Hef::create(hef_file);

  if (!hef_exp) {
    return hailort::make_unexpected(hef_exp.status());
  }

  hailort::Hef hef = hef_exp.release();

  hailort::Expected<hailort::NetworkGroupsParamsMap> configure_params =
    hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);

  if (!configure_params) {
    return hailort::make_unexpected(configure_params.status());
  }

  hailort::Expected<hailort::ConfiguredNetworkGroupVector> network_groups =
    this->vdevice->configure(hef, configure_params.value());

  if (!network_groups) {
    return hailort::make_unexpected(network_groups.status());
  }

  // Ensure exactly one network group was created
  if (network_groups->size() != 1) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "\nError: detection_inference: configure_network_group:\n";
      std::cerr << "Invalid amount of network groups" << std::endl << RESET;
    }

    return hailort::make_unexpected(HAILO_INTERNAL_FAILURE);
  }

  std::cout << BOLDBLUE << "Successfully configured network group" << std::endl << RESET;

  return std::move(network_groups->at(0));
}

void DetectionInference::setFrameReadyCallback(FrameReadyCallback callback)
{
  this->frameReadyCallback = std::move(callback);
}

hailo_status DetectionInference::stop()
{
  if (!this->hasDetectionStarted) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDYELLOW << "DetectionInference::stop():\n";
    std::cout << "Failed to stop detection inference. Detection inference has not started" << std::endl << RESET;
    return HAILO_SUCCESS;
  }
  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "\nStopping detection inference..." << std::endl << RESET;
  }

  this->hasDetectionStarted = false;

  for (auto &future : this->readFromHailoDeviceThreads) {
    future.wait();  // Wait for all inference threads to stop
  }

  this->readFromHailoDeviceThreads.clear();

  if (this->postprocessThread.valid()) {
    this->postprocessThread.wait(); // Wait for postprocessing thread to stop
  }

  this->readFromHailoDeviceThreads.clear();

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "\nDetection inference stopped\n" << std::endl << RESET;
  }

  return HAILO_SUCCESS;
}

hailo_status DetectionInference::start()
{
  if (this->hasDetectionStarted) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDYELLOW << "DetectionInference::start():\n";
    std::cout << "Detection inference has already started" << std::endl << RESET;
    return HAILO_SUCCESS;
  }

  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "\nDetection inference started\n" << std::endl << RESET; 
  }

  hailo_status status = HAILO_UNINITIALIZED;
  this->hasDetectionStarted = true;

  std::chrono::duration<double> total_time;
  std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

  status = this->startInference<uint8_t>(std::ref(this->vstreams.second));

  if (HAILO_SUCCESS != status) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: startDetection:\n";
      std::cerr << "Failed starting inference with status = " << status << std::endl << RESET;
    }
    return status;
  }

  std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
  total_time = t_end - t_start;
  
  {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << BOLDBLUE << "\nDetectionInference finished successfully\n";
    std::cout << "Total run time: " << total_time.count() << " seconds" << RESET << std::endl;
  }

  return HAILO_SUCCESS;
}

void DetectionInference::filterROI(HailoROIPtr roi)
{
  if (!roi->has_tensors()) {
    {
      std::lock_guard<std::mutex> lock(io_mutex);
      std::cout << BOLDYELLOW << "\nDetectionInference::filterROI():\n";
      std::cout << "No tensors! Nothing to process!" << std::endl << RESET;
    }
    return;
  }

  std::vector<HailoTensorPtr> tensors = roi->get_tensors();
  std::unordered_map<uint8_t, std::string> labels_map;

  if (tensors[0]->name().find("mobilenet") != std::string::npos) {
    labels_map = this->cocoNinetyLabels;
  }
  else {
    labels_map = this->cocoEightyLabels;
  }

  for (HailoTensorPtr tensor : tensors) {
    // Check if the tensor's name contains the substring "nms"
    if (std::regex_search(tensor->name(), std::regex("nms"))) {
      // Create a decoder object for non-maximum suppression using the tensor and label mapping
      HailoNMSDecode post(tensor, labels_map);
      // Decode the tensor into detection results
      std::vector<HailoDetection> detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
      // Add the decoded detections to the ROI
      hailo_common::add_detections(roi, detections);
    }
  }
}

#endif