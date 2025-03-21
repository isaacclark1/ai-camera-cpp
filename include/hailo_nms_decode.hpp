/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

 #include <vector>
 #include <string>
 #include <iostream>
 #include <regex>
 #include <unordered_map>
 
 // Include project-specific headers that define Hailo objects and YOLO utilities.
 #include "hailo_objects.hpp"
 #include "structures.hpp"
 #include "dataset_labels/coco_ninety.hpp"
 
 // Define default constants for maximum bounding boxes and detection threshold.
 static const int DEFAULT_MAX_BOXES = 100;
 static const float DEFAULT_THRESHOLD = 0.4f;
 
 // HailoNMSDecode class is used to decode the output tensor (after non-maximum suppression)
 // from a neural network into a vector of detection objects.
 class HailoNMSDecode
 {
 private:
     // Pointer to the tensor containing NMS output data.
     HailoTensorPtr _nms_output_tensor;
 
     // Mapping from class IDs (as uint8_t) to their string labels.
     std::unordered_map<uint8_t, std::string> labels_dict;
 
     // Detection threshold: Only detections with a score higher than this are considered.
     float _detection_thr;
 
     // Maximum number of bounding boxes to consider.
     uint _max_boxes;
 
     // Flag to decide whether to filter detections by their score.
     bool _filter_by_score;
 
     // The vstream info (metadata) associated with the tensor, including shape and quantization info.
     const hailo_vstream_info_t _vstream_info;
 
     // Dequantizes a bounding box structure from quantized values (e.g., uint16_t) to float32 values.
     // This converts the raw quantized output from the neural network to human-interpretable numbers.
     common::hailo_bbox_float32_t dequantize_hailo_bbox(const auto *bbox_struct)
     {
         common::hailo_bbox_float32_t dequant_bbox = {
             .y_min = _nms_output_tensor->fix_scale(bbox_struct->y_min),
             .x_min = _nms_output_tensor->fix_scale(bbox_struct->x_min),
             .y_max = _nms_output_tensor->fix_scale(bbox_struct->y_max),
             .x_max = _nms_output_tensor->fix_scale(bbox_struct->x_max),
             .score = _nms_output_tensor->fix_scale(bbox_struct->score)
         };
         return dequant_bbox;
     }
 
     // Converts a dequantized bounding box into a detection object and adds it to the _objects vector.
     // It applies a score filter if required.
     void parse_bbox_to_detection_object(auto dequant_bbox, uint32_t class_index, std::vector<HailoDetection> &_objects)
     {
         // Clamp the score between 0.0 and 1.0.
         float confidence = CLAMP(dequant_bbox.score, 0.0f, 1.0f);
         // If filtering by score is disabled or the detection meets the threshold, create a detection.
         if (!_filter_by_score || dequant_bbox.score > _detection_thr)
         {
             float32_t w, h = 0.0f;
             // Calculate the width and height of the bounding box.
             std::tie(w, h) = get_shape(&dequant_bbox);
             // Create a new HailoDetection object and add it to the _objects vector.
             _objects.push_back(HailoDetection(
                 HailoBBox(dequant_bbox.x_min, dequant_bbox.y_min, w, h), 
                 class_index, 
                 labels_dict[(unsigned char)class_index], 
                 confidence
             ));
         }
     }
 
     // Computes the width and height of a bounding box from the given bounding box structure.
     std::pair<float, float> get_shape(auto *bbox_struct)
     {
         float32_t w = static_cast<float32_t>(bbox_struct->x_max - bbox_struct->x_min);
         float32_t h = static_cast<float32_t>(bbox_struct->y_max - bbox_struct->y_min);
         return std::pair<float, float>(w, h);
     }
 
 public:
     // Constructor for HailoNMSDecode.
     // Initializes the decoder with the given output tensor, label mapping, detection threshold,
     // maximum number of boxes, and a flag indicating whether to filter by score.
     HailoNMSDecode(HailoTensorPtr tensor, std::unordered_map<uint8_t, std::string> &labels_dict, float detection_thr = DEFAULT_THRESHOLD, uint max_boxes = DEFAULT_MAX_BOXES, bool filter_by_score = false)
         : _nms_output_tensor(tensor),
           labels_dict(labels_dict),
           _detection_thr(detection_thr),
           _max_boxes(max_boxes),
           _filter_by_score(filter_by_score),
           _vstream_info(tensor->vstream_info())
     {
         // Ensure that the output tensor is of NMS type.
         // MODIFIED BY Isaac Clark 21/03/2025 to use the new constant HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS.
         //if (HAILO_FORMAT_ORDER_HAILO_NMS != _vstream_info.format.order) { !---- DEPRACATED - CORRECT CONSTANT IS NOW: HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS ---!
         if (HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS != _vstream_info.format.order) {
            std::cout << "Output tensor " << _nms_output_tensor->name() << " is not an NMS type" << std::endl;
            throw std::invalid_argument("Output tensor " + _nms_output_tensor->name() + " is not an NMS type");
         }
     };
 
     /**
      * Decode method:
      * This template function decodes the NMS tensor output into a vector of HailoDetection objects.
      * - T: The type of the quantized data (e.g., uint16_t).
      * - BBoxType: The type of the bounding box structure.
      */
     template <typename T, typename BBoxType>
     std::vector<HailoDetection> decode()
     {
         // Vector to hold the decoded detection objects.
         std::vector<HailoDetection> _objects;
         if (!_nms_output_tensor)
             return _objects;
 
         // Reserve space for the detections.
         _objects.reserve(_max_boxes);
         // Get a pointer to the beginning of the tensor's data.
         uint8_t *src_ptr = _nms_output_tensor->data();
         uint32_t actual_frame_size = 0;
 
         // Retrieve the number of classes and the maximum number of bounding boxes per class from the tensor metadata.
         uint32_t num_of_classes = _vstream_info.nms_shape.number_of_classes;
         uint32_t max_bboxes_per_class = _vstream_info.nms_shape.max_bboxes_per_class;
 
         // Loop over each class (starting from class 1 up to num_of_classes).
         for (uint32_t class_index = 1; class_index <= num_of_classes; class_index++)
         {
             // Read the number of bounding boxes for the current class.
             T bbox_count = *reinterpret_cast<const T *>(src_ptr + actual_frame_size);
 
             // If the count exceeds the maximum allowed per class, throw an error.
             if ((int)bbox_count > max_bboxes_per_class)
                 throw std::runtime_error(("Runtime error - Got more than the maximum bboxes per class in the nms buffer"));
 
             // If there are any bounding boxes for this class...
             if (bbox_count > 0)
             {
                 // Pointer to the start of the bounding box data for this class.
                 uint8_t *class_ptr = src_ptr + actual_frame_size + sizeof(bbox_count);
 
                 // Process each bounding box.
                 for (uint8_t box_index = 0; box_index < bbox_count; box_index++)
                 {
                     // Cast the data pointer to the bounding box structure type.
                     BBoxType *bbox_struct = (BBoxType *)(class_ptr + (box_index * sizeof(BBoxType)));
 
                     if (std::is_same<T, uint16_t>::value)
                     {
                         // If the data is quantized (e.g., uint16_t), dequantize the values first.
                         common::hailo_bbox_float32_t dequant_bbox = dequantize_hailo_bbox(bbox_struct);
                         // Convert the dequantized bounding box to a detection object.
                         parse_bbox_to_detection_object(dequant_bbox, class_index, _objects);
                     }
                     else
                     {
                         // If data is not quantized, directly convert the bounding box.
                         parse_bbox_to_detection_object(*bbox_struct, class_index, _objects);
                     }
                 }
             }
 
             // Update the offset in the buffer based on the size of the processed class data.
             T class_frame_size = static_cast<T>(sizeof(bbox_count) + bbox_count * sizeof(BBoxType));
             actual_frame_size += static_cast<uint32_t>(class_frame_size);
         }
 
         // Return the vector of decoded detection objects.
         return _objects;
     }
 };