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
#include <yolo_hailortpp.hpp>
#include "hailo_objects.hpp"
#include "FrameQueue.hpp"
#include <hailo/hailort.hpp>
#include <regex>
#include "dataset_labels/coco_eighty.hpp"
#include "dataset_labels/coco_ninety.hpp"
#include "hailo_nms_decode.hpp"
#include <stdexcept>

/**
 * Callback function that processes an OpenCV image (cv::Mat)
 * along with a vector of detection results (HailoDetectionPointer)
 */
using MatCallback = std::function<void(cv::Mat &, std::vector<HailoDetectionPtr> &)>;

/**
 * DetectionInference encapsulates the logic to perform detection inference on image frames
 * using the Hailo device.
 */
class DetectionInference
{
  private:
    std::mutex frameQueueMutex;
    std::mutex io_mutex; // Input/Output

    const bool QUANTIZED = true; // Reduce precision of numerical values in the neural network
    const hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO; // Automatically determine the format type for the HAILO device
    
    double frameHeight;
    double frameWidth;

    std::unique_ptr<hailort::VDevice> vdevice; // Hailo VDevice (virtual device) representing the hailo hardware
    // The loaded neural network that will be used for inference
    std::shared_ptr<hailort::ConfiguredNetworkGroup> networkGroup;
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams;
    std::unique_ptr<FrameQueue<cv::Mat>> frames;
    std::atomic<bool> hasDetectionStarted{false};
    MatCallback matCallback;

    /**
     * Setup virtual device, network group, and input/output video streams.
     * 
     * @param hef_file The compiled network file for the Hailo device.
     * @param frame_height The height of the frame.
     * @param frame_width The width of the frame.
     */
    DetectionInference(std::string hef_file, double frame_height, double frame_width);

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
     * @param frames The frames to add the detections to.
     * @param frame_height The height of the frame.
     * @param frame_width The width of the frame.
     */
    template <typename T>
    hailo_status startPostprocessing(std::vector<std::shared_ptr<FeatureData<T>>> &features);

    /**
     * Continuously read data from the inference engine.
     * 
     * @param output_vstream The output stream to read from.
     * @param feature The feature data to read into.
     */
    template <typename T>
    hailo_status readFromOutputDevice(hailort::OutputVStream &output_vstream, std::shared_ptr<FeatureData<T>> feature);

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
    void printNetworkBanner(
      std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams
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

  public:
    // Delete copy constructor and assignment operator to prevent copying
    DetectionInference(const DetectionInference &) = delete;
    DetectionInference& operator=(const DetectionInference &) = delete;

    /**
     * Static method to get the singleton instance.
     * 
     * @param hef_file The compiled network file for the Hailo device.
     * @param frame_height The desired height of the frame.
     * @param frame_width The desired width of the frame.
     */
    static DetectionInference& getInstance(std::string hef_file, double frame_height, double frame_width);

    ~DetectionInference() {}

    /**
     * Start the detection inference
     */
    hailo_status start();

    /**
     * Stop the detection infererence pipeline.
     */
    void stop();

    /**
     * Send an image frame to the inference engine.
     * 
     * @param frame The image frame.
     */
    void writeFrameToDevice(cv::Mat frame);

    /**
      * Set the callback function that is called when an OpenCV Mat and detections are ready.
      * 
      * @param callback The callback function.
      */
    void setMatCallback(MatCallback callback);
};

// ==============================
// 🔹 Constructor definition 🔹
// ==============================

DetectionInference::DetectionInference(std::string hef_file, double frame_height, double frame_width)
  : frames(std::make_unique<FrameQueue<cv::Mat>>(10)),
    frameHeight(frame_height),
    frameWidth(frame_width)
{
  hailort::Expected<std::unique_ptr<hailort::VDevice>> vdevice_exp = hailort::VDevice::create();

  if (!vdevice_exp) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: startDetection:\n";
      std::cerr << "Failed to create vdevice, status = " << vdevice_exp.status() << std::endl << RESET;
    }

    throw std::runtime_error("Failed to create vdevice with status: " + std::to_string(vdevice_exp.status()));
  }

  this->vdevice = vdevice_exp.release();
  hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> network_group_exp =
    configureNetworkGroup(hef_file);

  if (!network_group_exp) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: startDetection:\n";
      std::cerr << "Failed to configure network group: " << hef_file << std::endl << RESET;
    }

    throw std::runtime_error(
      "Failed to configure network group with status: " + std::to_string(network_group_exp.status())
    );
  }

  this->networkGroup = network_group_exp.release();
  hailort::Expected<std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>>>
    vstreams_exp = hailort::VStreamsBuilder::create_vstreams(*networkGroup, QUANTIZED, FORMAT_TYPE);
  
  if (!vstreams_exp) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: startDetection:\n";
      std::cerr << "Failed to create vstreams: " << vstreams_exp.status() << std::endl << RESET;
    }

    throw std::runtime_error("Failed to create vstreams with status: " + std::to_string(vstreams_exp.status()));
  }

  this->vstreams = vstreams_exp.release();
  this->printNetworkBanner(this->vstreams);
}


// ==================================
// 🔹 Member function Definitions 🔹
// ==================================

DetectionInference& DetectionInference::getInstance(
  std::string hef_file = "/home/isaac/projects/ai-camera-cpp/network-files/yolov8s.hef",
  double frame_height = 1080,
  double frame_width = 1920
)
{
  static std::mutex instanceMutex;
  std::lock_guard<std::mutex> lock(instanceMutex);

  static DetectionInference instance(hef_file, frame_height, frame_width);
  return instance;
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
    cv::Scalar colour(120, 120, 120);

    if (label == "person")            colour = cv::Scalar(50, 200, 0);    // Green
    else if (label == "car")          colour = cv::Scalar(200, 50, 0);    // Blue
    else if (label == "bicycle")      colour = cv::Scalar(0, 0, 200);     // Red
    else if (label == "dog")          colour = cv::Scalar(0, 180, 240);   // Yellow
    else if (label == "motorcycle")   colour = cv::Scalar(180, 180, 0);   // Teal
    else if (label == "bus")          colour = cv::Scalar(200, 0, 140);   // Purple
    else if (label == "truck")        colour = cv::Scalar(255, 140, 0);   // Blue

    // Draw the bounding box on the frame
    cv::rectangle(frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), colour, 2);

    // Round the most significant digit from the detection confidence to the nearest integer
    uint8_t rounded_confidence = static_cast<uint8_t>(((detection->get_confidence() * 100) / 10) + 0.5f);
    
    std::string text = label + " " + std::to_string(rounded_confidence) + "0%";

    int baseline = 0;
    float textSize = 1;
    float textThickness = 2;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, textSize, textThickness, &baseline);

    // Define the position for the text (above the bounding box)
    cv::Point text_position(xmin, ymin - 10);

    // Draw a filled rectangle as background for the text
    cv::rectangle(
      frame,
      cv::Point(text_position.x, text_position.y - text_size.height - baseline),
      cv::Point(text_position.x + text_size.width, text_position.y + baseline),
      colour,
      cv::FILLED
    );

    // Put the label text over the background rectangle
    cv::putText(frame, text, text_position, cv::FONT_HERSHEY_SIMPLEX, textSize, cv::Scalar(255, 255, 255), textThickness);
  }
}

template <typename T>
hailo_status DetectionInference::startPostprocessing(std::vector<std::shared_ptr<FeatureData<T>>> &features)
{
  // Sort feature data by tensor size
  std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

  {
    std::lock_guard<std::mutex> lock(this->io_mutex);
    std::cout << BOLDBLUE << "Starting postprocessing" << std::endl << RESET;
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
      first_frame = this->frames->front();
    }

    // Resize the first frame to the original dimensions
    cv::resize(first_frame, first_frame, cv::Size((int)this->frameWidth, (int)this->frameHeight));

    // Draw the detection results on the frame
    this->drawDetections(first_frame, detections);

    // Call the callback function if it is set
    if (this->matCallback != nullptr) {
      this->matCallback(first_frame, detections);
    }

    // Release the current frame and remove it from the queue
    first_frame.release();

    // Remove the first frame
    {
      std::lock_guard<std::mutex> lock(this->frameQueueMutex);
      this->frames->dequeue();
    }
  }

  return HAILO_SUCCESS;
}

template <typename T>
hailo_status DetectionInference::readFromOutputDevice(
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
        std::lock_guard<std::mutex> lock(this->io_mutex);
        std::cerr << BOLDRED << "Error: detection_inference: read_all:\n";
        std::cerr << "Failed reading with status = " << status << std::endl << RESET;
      }

      return status;
    }
  }

  return HAILO_SUCCESS;
}

void DetectionInference::writeFrameToDevice(cv::Mat frame)
{
  hailort::InputVStream &input_vstream = this->vstreams.first[0];

  hailo_3d_image_shape_t input_shape = input_vstream.get_info().shape;
  int height = input_shape.height;
  int width = input_shape.width;

  // Resize the input frame to the dimensions required by the network
  cv::resize(frame, frame, cv::Size(width, height), 1);

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
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cout << BOLDYELLOW << "DetectionInference::writeFrame:\n";
      std::cout << "Frame queue is empty" << std::endl << RESET;
    }
    return;
  }

  // Verify that the front frame is valid
  if (frontFrame.empty() || frontFrame.data == nullptr) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cout << BOLDYELLOW << "DetectionInference::writeFrame:\n";
      std::cout << "Invalid frame" << std::endl << RESET;
    }
    return;
  }
  
  hailo_status write_status = input_vstream.write(hailort::MemoryView(
    frontFrame.data,
    input_vstream.get_frame_size()
  ));

  if (write_status != HAILO_SUCCESS) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cout << BOLDYELLOW << "DetectionInference::writeFrame:\n";
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
        std::lock_guard<std::mutex> lock(this->io_mutex);
        std::cerr << BOLDRED << "Error: detection_inference: run_inference\n";
        std::cerr << "Failure creating feature with status = " << status << std::endl << RESET;
      }
      return status;
    }

    features.emplace_back(feature);
  }

  std::vector<std::future<hailo_status>> output_threads;
  output_threads.reserve(output_vstream_size);

  for (size_t i = 0; i < output_vstream_size; i++) {
    output_threads.emplace_back(
      std::async(
        std::launch::async,
        &DetectionInference::readFromOutputDevice<T>,
        this,
        std::ref(output_vstream[i]),
        features[i]
      )
    );
  }

  auto postprocess_thread(
    std::async(std::launch::async,
      &DetectionInference::startPostprocessing<T>,
      this,
      std::ref(features)
    )
  );

  // Wait for all read threads to complete
  for (size_t i = 0; i < output_threads.size(); i++) {
    status = output_threads[i].get();
  }

  auto postprocess_status = postprocess_thread.get(); // Wait for postprocessing thread to complete

  if (HAILO_SUCCESS != status) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: detection_inference: run_inference:\n";
      std::cerr << "Read failed with status: " << status << std::endl << RESET;
    }
    return status;
  }

  if (HAILO_SUCCESS != postprocess_status) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: detection_inference: run_inference:\n";
      std::cerr << "Post-processing failed with status " << postprocess_status << std::endl << RESET;
    }
    return postprocess_status;
  }


  {
    std::lock_guard<std::mutex> lock(this->io_mutex);
    std::cout << BOLDBLUE << "Inference finished successfully" << RESET << std::endl;
  }

  return HAILO_SUCCESS;
}

void DetectionInference::printNetworkBanner(
  std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams
)
{
  std::lock_guard<std::mutex> lock(this->io_mutex);

  std::cout << BOLDMAGENTA << "\n-------------------------------------------------\n";
  std::cout << "     Network  Name                                     \n";
  std::cout << "-------------------------------------------------\n";

  for (auto const &value : vstreams.first) {
    std::cout << "     IN:  " << value.name() << "\n";
  }

  std::cout << "-------------------------------------------------\n";

  for (auto const &value : vstreams.second) {
    std::cout << "     OUT: " << value.name() << "\n";
  }

  std::cout << "-------------------------------------------------\n" << std::endl << RESET;
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
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: detection_inference: configure_network_group:\n";
      std::cerr << "Invalid amount of network groups" << std::endl << RESET;
    }

    return hailort::make_unexpected(HAILO_INTERNAL_FAILURE);
  }

  std::cout << BOLDBLUE << "Configured network group" << std::endl << RESET;

  return std::move(network_groups->at(0));
}

void DetectionInference::setMatCallback(MatCallback callback)
{
  this->matCallback = std::move(callback);
}

void DetectionInference::stop()
{
  this->hasDetectionStarted = false;
}

hailo_status DetectionInference::start()
{
  {
    std::lock_guard<std::mutex> lock(this->io_mutex);
    std::cout << BOLDBLUE << "Detection inference started" << std::endl << RESET; 
  }

  hailo_status status = HAILO_UNINITIALIZED;
  this->hasDetectionStarted = true;

  std::chrono::duration<double> total_time;
  std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

  status = this->startInference<uint8_t>(std::ref(this->vstreams.second));

  if (HAILO_SUCCESS != status) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cerr << BOLDRED << "Error: DetectionInference: startDetection:\n";
      std::cerr << "Failed starting inference with status = " << status << std::endl << RESET;
    }
    return status;
  }

  std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
  total_time = t_end - t_start;
  
  {
    std::lock_guard<std::mutex> lock(this->io_mutex);
    std::cout << BOLDBLUE << "\nApplication run finished successfully\n";
    std::cout << "Total application run time: " << (double)total_time.count() << " seconds" << RESET << std::endl;
  }

  // Clear input and output vstreams
  this->vstreams.first.clear();
  this->vstreams.second.clear();

  this->networkGroup.reset();
  this->vdevice.reset();

  {
    std::lock_guard<std::mutex> lock(this->io_mutex);
    std::cout << BOLDBLUE << "DetectionInference stopped and resources cleaned up." << std::endl << RESET;
  }

  return HAILO_SUCCESS;
}

void DetectionInference::filterROI(HailoROIPtr roi)
{
  if (!roi->has_tensors()) {
    {
      std::lock_guard<std::mutex> lock(this->io_mutex);
      std::cout << BOLDYELLOW << "No tensors! Nothing to process!" << std::endl << RESET;
    }
    return;
  }

  std::vector<HailoTensorPtr> tensors = roi->get_tensors();
  std::map<uint8_t, std::string> labels_map;

  if (tensors[0]->name().find("mobilenet") != std::string::npos) {
    labels_map = common::coco_ninety_classes;
  }
  else {
    labels_map = common::coco_eighty_classes;
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