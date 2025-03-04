#include <DetectionInference.hpp>

#include <chrono>                        // For timing measurements
#include <future>                        // For async operations
#include <iostream>
#include <mutex>                         // Lock management
#include <opencv2/core/matx.hpp> // For matrix operations
#include <opencv2/highgui.hpp>   // For high-level GUI functions (displaying images)
#include <opencv2/imgcodecs.hpp> // For image encoding/decoding
#include <opencv2/opencv.hpp>    // Main OpenCV header
#include <thread>                        // For running concurrent threads

#include <common.h>
#include <hailo_objects.hpp>
#include <yolo_hailortpp.hpp>

constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;

static std::mutex mutex;                             // Mutex for synchronising access to shared resources
static std::atomic<bool> hasDetectionStarted{false}; // Has processing started?
static MatCallback matCallback;                      // Called when frames and detections are ready


// Convert vstream info into a string for logging
static std::string vstreamInfoToString(hailo_vstream_info_t vstream_info)
{
  std::string result = vstream_info.name;
  result += " (";
  result += std::to_string(vstream_info.shape.height);
  result += ", ";
  result += std::to_string(vstream_info.shape.width);
  result += ", ";
  result += std::to_string(vstream_info.shape.features);
  result += ")";
  return result;
}

// Function to generate a random colour using OpenCV's RNG
static cv::Scalar getRandomColour()
{
  cv::RNG rng(cv::getTickCount());
  int red = rng.uniform(0, 256);
  int green = rng.uniform(0, 256);
  int blue = rng.uniform(0, 256);
  return cv::Scalar(red, green, blue);
}

// Global map to store a consistent random colour for each detection label
static std::map<std::string, cv::Scalar> labelColours;

// Initialise label colors for each detection if not already set
static void initialiseColours(const std::vector<HailoDetectionPtr> &detections)
{
  for (const auto &detection : detections) {
    std::string label = detection->get_label();

    if (labelColours.find(label) == labelColours.end()) {
      labelColours[label] = getRandomColour();
    }
  }
}

// Main drawing function to overlay detection results on a frame
void drawDetections(cv::Mat &frame, const std::vector<HailoDetectionPtr> &detections, int original_width, int original_height)
{
  // Ensure each label has a corresponding random colour
  initialiseColours(detections);

  // Iterate through each detection
  for (const auto &detection : detections) {
    // Skip detections with zero confidence
    if (detection->get_confidence() == 0) {
      continue;
    }

    // Get the bounding box (normalised values) of the detection
    HailoBBox bbox = detection->get_bbox();

    // Convert normalised coordinates to absolute pixel coordinates
    float xmin = bbox.xmin() * static_cast<float>(original_width);
    float ymin = bbox.ymin() * static_cast<float>(original_height);
    float xmax = bbox.xmax() * static_cast<float>(original_width);
    float ymax = bbox.ymax() * static_cast<float>(original_height);

    // Retrieve the assigned colour for this detection label
    std::string label = detection->get_label();
    cv::Scalar colour = labelColours[label];

    // Draw the bounding box on the frame
    cv::rectangle(frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), colour, 2);

    // Prepare the label text with confidence percentage
    std::string text = label + " " + std::to_string(static_cast<int>(detection->get_confidence() * 100)) + "%";

    // Calculate text size and baseline
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

// Template function to perform post-processing on inference results.
// Processes feature data, runs filtering, generates detections, draws them and calls the callback.
template <typename T>
static hailo_status post_processing_all(
  std::vector<std::shared_ptr<FeatureData<T>>> &features,
  FrameQueue<cv::Mat> &frames,
  double original_height,
  double original_width
)
{
  // Sort feature data by tensor size
  std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

  mutex.lock();
  std::cout << YELLOW << "\n-I- Starting postprocessing\n" << std::endl << RESET;
  mutex.unlock();

  // Process frames while the system is running
  while (hasDetectionStarted) {
    // Create a region of interest (ROI) that covers the whole image
    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

    // Add tensors to the ROI from each feature
    for (uint i = 0; i < features.size(); i++) {
      roi->add_tensor(std::make_shared<HailoTensor>(
        reinterpret_cast<T *>(features[i]->m_buffers.get_read_buffer().data()), features[i]->m_vstream_info
      ));
    }

    // Filter detections using the ROI
    filter(roi);

    // Release the read buffers for each feature
    for (auto &feature : features) {
      feature->m_buffers.release_read_buffer();
    }

    // Get detection results from the ROI
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    // Get the frame from the front of the queue
    mutex.lock();
    cv::Mat first_frame = frames.front();
    mutex.unlock();

    // Resize the first frame to the original dimensions
    cv::resize(first_frame, first_frame, cv::Size((int)original_width, (int)original_height), 1);

    // Draw the detection results on the frame
    drawDetections(first_frame, detections, original_width, original_height);

    // Call the callback function if it is set
    if (matCallback != nullptr) {
      matCallback(first_frame, detections);
    }

    // Release the current frame and remove it from the vector
    first_frame.release();

    // Remove the first frame
    mutex.lock();
    frames.dequeue();
    mutex.unlock();
  }

  return HAILO_SUCCESS;
}

// Template function to continuously read data from an output stream.
// It reads data into feature buffers and releases them after reading.
template <typename T>
static hailo_status read_all(hailort::OutputVStream &output_vstream, std::shared_ptr<FeatureData<T>> feature)
{
  while (hasDetectionStarted) {
    // Get a reference to the feature's write buffer
    std::vector<T> &buffer = feature->m_buffers.get_write_buffer();
    // Read data from the output stream into the buffer
    hailo_status status = output_vstream.read(hailort::MemoryView(buffer.data(), buffer.size()));
    // Release the write bufer once the read is complete
    feature->m_buffers.release_write_buffer();

    if (HAILO_SUCCESS != status) {
      std::cerr << "Error: detection_inference: read_all:\n";
      std::cerr << "Failed reading with status = " << status << std::endl;
      return status;
    }
  }

  return HAILO_SUCCESS;
}

// Function to write frames to an input stream.
// It captures frames from a video source (using OpenCV), resizes them, and writes to the stream.
// static hailo_status write_all(hailort::InputVStream &input_vstream, std::vector<cv::Mat> &frames)
// {
//   mutex.lock();
//   std::cout << CYAN << "-I- Started write thread: " << vstreamInfoToString(input_vstream.get_info()) << std::endl << RESET;
//   mutex.unlock();

//   hailo_status status = HAILO_SUCCESS;
//   auto input_shape = input_vstream.get_info().shape;
//   int height = input_shape.height;
//   int width = input_shape.width;

//   cv::VideoCapture capture;  // OpenCV video capture device
//   cv::Mat original_frame;

//   // Continuously capture frames while the system is running
//   while (hasDetectionStarted) {
//     std::cout << "Writing frames to input_vstream" << std::endl;

//     // Capture a frame and copy into original_frame
//     capture >> original_frame;

//     if (original_frame.empty()) {
//       std::cout << "Captured empty frame" << std::endl;
//       break;
//     }

//     std::cout << "Captured frame" << std::endl;

//     // Resize the captured rame to match the expected input dimensions
//     cv::resize(original_frame, original_frame, cv::Size(width, height), 1);

//     std::cout << "Resized frame" << std::endl;

//     mutex.lock();
//     frames.push_back(original_frame);
//     mutex.unlock();

//     std::cout << "Added frame to vector" << std::endl;

//     // Write the frame to the input stream
//     input_vstream.write(hailort::MemoryView(
//       frames[frames.size() - 1].data,
//       input_vstream.get_frame_size()
//     ));

//     std::cout << "Wrote frame to input_vstream" << std::endl;

//     if (HAILO_SUCCESS != status) {
//       return status;
//     }

//     original_frame.release();
//   }

//   capture.release();
//   return HAILO_SUCCESS;
// }

// Member function: Write a frame into the inference pipeline.
// It resizes the input frame and writes it into the input video stream.
void DetectionInference::writeFrame(cv::Mat original_frame)
{
  hailort::InputVStream &input_vstream = vstreams.first[0];

  hailo_3d_image_shape_t input_shape = input_vstream.get_info().shape;
  int height = input_shape.height;
  int width = input_shape.width;

  // Resize the input frame to the dimensions required by the network
  cv::resize(original_frame, original_frame, cv::Size(width, height), 1);
  
  mutex.lock();
  frames->enqueue(original_frame);
  mutex.unlock();

  cv::Mat frontFrame;

  mutex.lock();
  bool isFrameQueueEmpty = frames->isEmpty();

  if (!isFrameQueueEmpty) {
    frontFrame = frames->front();
  }
  
  mutex.unlock();

  if (isFrameQueueEmpty) {
    std::cout << "DetectionInference::writeFrame:\n";
    std::cout << "Frame queue is empty" << std::endl;
    return;
  }

  // Verify that the front frame is valid
  if (frontFrame.empty() || frontFrame.data == nullptr) {
    std::cout << "DetectionInference::writeFrame:\n";
    std::cout << "Invalid frame" << std::endl;
    return;
  }

  // Write the frame data to the input stream
  hailo_status write_status = input_vstream.write(hailort::MemoryView(
    frontFrame.data,
    input_vstream.get_frame_size()
  ));

  if (write_status != HAILO_SUCCESS) {
    std::cout << BOLDRED << "DetectionInference::writeFrame:\n";
    std::cout << "Status: " << write_status << " -> Failed to write frame to input stream" << std::endl << RESET;
    return;
  }
}

// Template function to create a FeatureData object from vstream info and frame size.
// This sets up feature extraction parameters.
template <typename T>
static hailo_status create_feature(
  hailo_vstream_info_t vstream_info,
  size_t output_frame_size,
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

// Template function to run inference.
// It sets up feature extraction, starts read threads for output streams,
// and a postprocessing thread, then waits for all threads to finish.
template <typename T>
static hailo_status run_inference(
  std::vector<hailort::InputVStream> &input_vstream,
  std::vector<hailort::OutputVStream> &output_vstream,
  double original_height,
  double original_width,
  FrameQueue<cv::Mat> &frames
)
{
  std::cout << BOLDCYAN << "Running inference" << std::endl << RESET;

  hailo_status status = HAILO_UNINITIALIZED;
  auto output_vstream_size = output_vstream.size();

  // Create feature data for each output stream
  std::vector<std::shared_ptr<FeatureData<T>>> features;
  features.reserve(output_vstream_size);

  for (size_t i = 0; i < output_vstream_size; i++) {
    std::shared_ptr<FeatureData<T>> feature(nullptr);
    hailo_status status = create_feature(output_vstream[i].get_info(), output_vstream[i].get_frame_size(), feature);

    if (HAILO_SUCCESS != status) {
      std::cerr << BOLDRED << "Error: detection_inference: run_inference\n";
      std::cerr << "Failure creating feature with status = " << status << std::endl << RESET;
      return status;
    }

    features.emplace_back(feature);
  }

  // Create write thread - Old code - see DetectionInference::writeFrame
  // auto input_thread(std::async(write_all, std::ref(input_vstream[0]), std::ref(frames)));

  // Create and read threads for each output stream
  std::vector<std::future<hailo_status>> output_threads;
  output_threads.reserve(output_vstream_size);

  for (size_t i = 0; i < output_vstream_size; i++) {
    output_threads.emplace_back(std::async(read_all<T>, std::ref(output_vstream[i]), features[i]));
  }

  // Create a postprocessing thread to process features and frames
  auto postprocess_thread(std::async(post_processing_all<T>, std::ref(features), std::ref(frames), original_height, original_width));

  // Wait for all read threads to complete
  for (size_t i = 0; i < output_threads.size(); i++) {
    status = output_threads[i].get();
  }

  // auto input_status = input_thread.get();
  auto postprocess_status = postprocess_thread.get();

  // if (HAILO_SUCCESS != input_status) {
  //   std::cerr << "Error: detection_inference: run_inference():\n";
  //   std::cerr << "Write thread failed with status: " << input_status << std::endl;
  //   return input_status;
  // }

  if (HAILO_SUCCESS != status) {
    std::cerr << "Error: detection_inference: run_inference:\n";
    std::cerr << "Read failed with status: " << status << std::endl;
    return status;
  }

  if (HAILO_SUCCESS != postprocess_status) {
    std::cerr << "Error: detection_inference: run_inference:\n";
    std::cerr << "Post-processing failed with status " << postprocess_status << std::endl;
    return postprocess_status;
  }

  std::cout << BOLDBLUE << "\n-I- Inference finished successfully" << RESET << std::endl;
  return HAILO_SUCCESS;
}

// Utility function to print network banner information using vstream info
static void print_net_banner(
  std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams
)
{
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
  std::cout << BOLDMAGENTA << "-I-  Network  Name                                     " << std::endl << RESET;
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
  for (auto const &value : vstreams.first) {
    std::cout << MAGENTA << "-I-  IN:  " << value.name() << std::endl << RESET;
  }
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
  for (auto const &value : vstreams.second) {
    std::cout << MAGENTA << "-I-  OUT: " << value.name() << std::endl << RESET;
  }
  std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

// Function to configure the network group for inference using a given HEF file
static hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> configure_network_group(
  hailort::VDevice &vdevice,
  std::string yolo_hef
)
{
  // Create a HEF object from the provided file
  auto hef_exp = hailort::Hef::create(yolo_hef);

  if (!hef_exp) {
    return hailort::make_unexpected(hef_exp.status());
  }

  auto hef = hef_exp.release();

  // Create configuration parameters for PCIe interface
  auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);

  if (!configure_params) {
    return hailort::make_unexpected(configure_params.status());
  }

  // Configure the VDevice using the HEF and configuration parameters
  auto network_groups = vdevice.configure(hef, configure_params.value());

  if (!network_groups) {
    return hailort::make_unexpected(network_groups.status());
  }

  // Ensure exactly one network group was created
  if (network_groups->size() != 1) {
    std::cerr << "Error: detection_inference: configure_network_group:\n";
    std::cerr << "Invalid amount of network groups" << std::endl;
    return hailort::make_unexpected(HAILO_INTERNAL_FAILURE);
  }

  return std::move(network_groups->at(0));
}

// Utility function to parse command line options
std::string getCmdOption(int argc, char *argv[], const std::string &option)
{
  std::string cmd;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (0 == arg.find(option, 0)) {
      std::size_t found = arg.find("=", 0) + 1;
      cmd = arg.substr(found);
      return cmd;
    }
  }

  return cmd;
}

// Set the global MatCallback used by the detection inference
void DetectionInference::setMatCallback(MatCallback callback)
{
  matCallback = std::move(callback);
}

// Stop the detection inference by setting the global flag
void DetectionInference::stop()
{
  hasDetectionStarted = false;
}

// Start the detection inference process
int DetectionInference::startDetection(double original_height, double original_width)
{
  std::cout << "Detection inference started" << std::endl;

  hailo_status status = HAILO_UNINITIALIZED;
  hasDetectionStarted = true;

  // Record the start time
  std::chrono::duration<double> total_time;
  std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

  // Path to the HEF file (compiled network file for YOLO)
  std::string yolo_hef = "/home/isaac/projects/ai-camera-cpp/network-files/yolov8m.hef";

  // Create a VDevice (virtual device) for inference
  auto vdevice_exp = hailort::VDevice::create();

  if (!vdevice_exp) {
    std::cerr << "Error: DetectionInference: startDetection:\n";
    std::cerr << "Failed to create vdevice, status = " << vdevice_exp.status() << std::endl;
    return vdevice_exp.status();
  }

  auto vdevice = vdevice_exp.release();

  // Configure the network group using the HEF file
  auto network_group_exp = configure_network_group(*vdevice, yolo_hef);

  if (!network_group_exp) {
    std::cerr << "Error: DetectionInference: startDetection:\n";
    std::cerr << "Failed to configure network group: " << yolo_hef << std::endl;
    return network_group_exp.status();
  }

  auto network_group = network_group_exp.release();

  // Build the input and output video streams for the network
  auto vstreams_exp = hailort::VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);

  if (!vstreams_exp) {
    std::cerr << "Error: DetectionInference: startDetection:\n";
    std::cerr << "Failed to create vstreams: " << vstreams_exp.status() << std::endl;
    return vstreams_exp.status();
  }

  this->vstreams = vstreams_exp.release();

  // Print network information banner
  print_net_banner(vstreams);

  // Run the inference pipeline: read from output streams, process features and postprocess results
  status = run_inference<uint8_t>(
    std::ref(vstreams.first),
    std::ref(vstreams.second),
    original_height,
    original_width,
    *frames
  );

  if (HAILO_SUCCESS != status) {
    std::cerr << "Error: DetectionInference: startDetection:\n";
    std::cerr << "Failed running inference with status = " << status << std::endl;
    return status;
  }

  // Record the end time and calculate total processing time
  std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
  total_time = t_end - t_start;
  std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
  std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;

  // Clear video streams
  vstreams.first.clear();
  vstreams.second.clear();

  // Release resources: network group and VDevice
  m_network_group.reset();
  m_vdevice.reset();

  std::cout << "DetectionInference stopped and resources cleaned up." << std::endl;
  return HAILO_SUCCESS;
}