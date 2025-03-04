#ifndef DETECTION_INFERENCE_HPP
#define DETECTION_INFERENCE_HPP

#include <condition_variable>         // Used for thread synchronisation
#include <mutex>                      // Mutex for locking shared resources
#include <opencv2/opencv.hpp>         // OpenCV core functionality
#include <queue>
#include <thread>                     // Thread support for running concurrent tasks

#include <hailo/hailort.hpp> // Hailo runtime API
#include "hailo_objects.hpp" // Definitions for hailo objects
#include "FrameQueue.hpp"    // Queue to hold frames

/**
 * Define a type alias for a callback function that processes an OpenCV image (cv::Mat)
 * along with a vector of detection results (HailoDetectionPointer)
 */
using MatCallback = std::function<void(cv::Mat &, std::vector<HailoDetectionPtr> &)>;

/**
 * DetectionInference encapsulates the logic to perform detection inference on image frames
 * using the Hailo device and network group
 */
class DetectionInference
{
  private:
    // Shared pointer to a Hailo VDevice representing the hailo hardware
    std::shared_ptr<hailort::VDevice> m_vdevice;

    /**
     * Shared pointer to a configured network group; this is the loaded neural network
     * that will be used for inference
     */
    std::shared_ptr<hailort::ConfiguredNetworkGroup> m_network_group;

    /**
     * A pair holding the input and output video streams for the network.
     * The first element is a vector of input streams, and the second element is
     * a vector of output streams
     */
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams;

    // A vector of OpenCV Mat objects that store image frames
    std::unique_ptr<FrameQueue<cv::Mat>> frames;

  public:
    DetectionInference() : frames(std::make_unique<FrameQueue<cv::Mat>>(10)) {};

    // Inititalise and start the detection inference
    int startDetection(double original_height, double original_width);

    // Send an image frame to the inference engine
    void writeFrame(cv::Mat original_frame);

    // Stop the ongoing detection inference
    void stop();

    /**
      * Set the callback function that is called when an OpenCV Mat and detections are ready.
      * This allows the user of the class to specify custom processing for the output frame and
      * its detections.
      */
    void setMatCallback(MatCallback callback);
};

#endif