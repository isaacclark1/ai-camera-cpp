#include "hailo_nms_decode.hpp"   // Provdes functionality to decode NMS (non-max supression) output
#include "yolo_hailortpp.hpp"     // Contans YOLO (You Only Look Once) related inference utilities

#include "dataset_labels/coco_eighty.hpp" // Label mapping for an 80-class COCO dataset
#include "dataset_labels/coco_ninety.hpp" // Label mapping for a 90-class COCO dataset

#include <regex>

/**
 * The filter function processes a Region Of Interest by decoding tensor data
 * using Non-Maximum Suppression (NMS) and then adding the resulting detections to the ROI.
 */
void filter(HailoROIPtr roi, void *params_void_ptr)
{
  // If the ROI does not contain any tensors, there is nothing to process, so return
  if (!roi->has_tensors()) {
    std::cout << "No tensors! Nothing to process!" << std::endl;
    return;
  }

  // Retrieve the tensors associated with the ROI
  std::vector<HailoTensorPtr> tensors = roi->get_tensors();
  // Map to hold label mappings
  std::map<uint8_t, std::string> labels_map;

  // Choose the correct set of label mappings based on the first tensor's name.
  // If it contains "mobilenet", use the 90-class mapping; otherwise, usethe 80-class mapping
  if (tensors[0]->name().find("mobilenet") != std::string::npos) {
    labels_map = common::coco_ninety_classes;
  }
  else {
    labels_map = common::coco_eighty_classes;
  }

  // Loop over each tensor in the ROI
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