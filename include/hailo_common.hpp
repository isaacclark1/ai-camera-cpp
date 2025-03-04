/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

 #pragma once
 #include "hailo_objects.hpp"
 
 // Undefine __BEGIN_DECLS and __END_DECLS in case they were defined before.
 #undef __BEGIN_DECLS
 #undef __END_DECLS
 
 #ifdef __cplusplus
 // If compiling as C++, use extern "C" to prevent name mangling for C declarations.
 #define __BEGIN_DECLS \
     extern "C"        \
     {
 #define __END_DECLS }
 #else
 // If compiling as C, these macros are empty.
 #define __BEGIN_DECLS /* empty */
 #define __END_DECLS   /* empty */
 #endif
 
 // Begin the hailo_common namespace to group common utility functions.
 namespace hailo_common
 {
     // --------------------------------------------------------------------
     // Add a single HailoObject to a given ROI.
     inline void add_object(HailoROIPtr roi, HailoObjectPtr obj)
     {
         roi->add_object(obj);
     }
 
     // --------------------------------------------------------------------
     // Add a vector of HailoObjects to a given ROI.
     inline void add_objects(HailoROIPtr roi, std::vector<HailoObjectPtr> objects)
     {
         for (HailoObjectPtr obj : objects)
         {
             add_object(roi, obj);
         }
     }
 
     // --------------------------------------------------------------------
     // Create and add a HailoClassification object to the ROI.
     // 'type' and 'label' describe the classification; 'confidence' is the detection score.
     // 'class_id' defaults to NULL_CLASS_ID if not provided.
     inline void add_classification(HailoROIPtr roi, std::string type, std::string label, float confidence, int class_id = NULL_CLASS_ID)
     {
         add_object(roi,
                    std::make_shared<HailoClassification>(type, class_id, label, confidence));
     }
 
     // --------------------------------------------------------------------
     // Create and add a HailoDetection object to the ROI.
     // Constructs a detection from the given bounding box, label, confidence, and class id.
     // Sets the detection's scaling bounding box to match the ROI's bounding box.
     // Returns the created detection pointer.
     inline HailoDetectionPtr add_detection(HailoROIPtr roi, HailoBBox bbox, std::string label, float confidence, int class_id = NULL_CLASS_ID)
     {
         HailoDetectionPtr detection = std::make_shared<HailoDetection>(bbox, class_id, label, confidence);
         detection->set_scaling_bbox(roi->get_bbox());
         add_object(roi, detection);
         return detection;
     }
 
     // --------------------------------------------------------------------
     // Add multiple HailoDetection objects (given as a vector of HailoDetection)
     // to the ROI by converting each into a shared pointer.
     inline void add_detections(HailoROIPtr roi, std::vector<HailoDetection> detections)
     {
         for (auto det : detections)
         {
             add_object(roi, std::make_shared<HailoDetection>(det));
         }
     }
 
     // --------------------------------------------------------------------
     // Add multiple HailoDetection pointers to the ROI.
     // For each detection, set its scaling bounding box and add it to the ROI.
     inline void add_detection_pointers(HailoROIPtr roi, std::vector<HailoDetectionPtr> detections)
     {
         for (auto det : detections)
         {
             det->set_scaling_bbox(roi->get_bbox());
             roi->add_object(det);
         }
     }
 
     // --------------------------------------------------------------------
     // Remove a set of HailoObjects from the ROI.
     inline void remove_objects(HailoROIPtr roi, std::vector<HailoObjectPtr> objects)
     {
         for (HailoObjectPtr obj : objects)
         {
             roi->remove_object(obj);
         }
     }
 
     // --------------------------------------------------------------------
     // Remove a set of HailoDetection objects from the ROI.
     inline void remove_detections(HailoROIPtr roi, std::vector<HailoDetectionPtr> objects)
     {
         for (HailoObjectPtr obj : objects)
         {
             roi->remove_object(obj);
         }
     }
 
     // --------------------------------------------------------------------
     // Check if the ROI has any classification objects of a specific type.
     inline bool has_classifications(HailoROIPtr roi, std::string classification_type)
     {
         for (auto obj : roi->get_objects_typed(HAILO_CLASSIFICATION))
         {
             // Cast the object to a HailoClassification pointer.
             HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
             // Compare the classification type; if a match is found, return true.
             if (classification_type.compare(classification->get_classification_type()) == 0)
             {
                 return true;
             }
         }
         return false;
     }
 
     // --------------------------------------------------------------------
     // Retrieve all HailoDetection objects from the ROI as a vector of shared pointers.
     inline std::vector<HailoDetectionPtr> get_hailo_detections(HailoROIPtr roi)
     {
         // Get all objects in the ROI that are of detection type.
         std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_DETECTION);
         std::vector<HailoDetectionPtr> detections;
 
         for (auto obj : objects)
         {
             // Cast each object to HailoDetectionPtr and add it to the detections vector.
             detections.emplace_back(std::dynamic_pointer_cast<HailoDetection>(obj));
         }
         return detections;
     }
 
     // --------------------------------------------------------------------
     // Retrieve all HailoTileROI objects from the ROI.
     inline std::vector<HailoTileROIPtr> get_hailo_tiles(HailoROIPtr roi)
     {
         // Get objects of type HAILO_TILE.
         std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_TILE);
         std::vector<HailoTileROIPtr> tiles;
 
         for (auto obj : objects)
         {
             // Cast each object to HailoTileROI and add it to the tiles vector.
             tiles.emplace_back(std::dynamic_pointer_cast<HailoTileROI>(obj));
         }
         return tiles;
     }
 
     // --------------------------------------------------------------------
     // Retrieve HailoClassification objects from the ROI.
     // Optionally filter by a specific classification type.
     inline std::vector<HailoClassificationPtr> get_hailo_classifications(HailoROIPtr roi, std::string classification_type = "")
     {
         std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_CLASSIFICATION);
         std::vector<HailoClassificationPtr> classifications;
 
         for (auto obj : objects)
         {
             HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
             // If no specific type is requested or if the type matches, add it to the vector.
             if (classification_type.empty() || classification_type.compare(classification->get_classification_type()) == 0)
             {
                 classifications.emplace_back(std::dynamic_pointer_cast<HailoClassification>(obj));
             }
         }
         return classifications;
     }
 
     // --------------------------------------------------------------------
     // Retrieve all HailoUniqueID objects from the ROI.
     inline std::vector<HailoUniqueIDPtr> get_hailo_unique_id(HailoROIPtr roi)
     {
         std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_UNIQUE_ID);
         std::vector<HailoUniqueIDPtr> unique_ids;
 
         for (auto obj : objects)
         {
             unique_ids.emplace_back(std::dynamic_pointer_cast<HailoUniqueID>(obj));
         }
         return unique_ids;
     }
 
     // --------------------------------------------------------------------
     // Retrieve all HailoUniqueID objects with a specific mode from the ROI.
     inline std::vector<HailoUniqueIDPtr> get_hailo_unique_id_by_mode(HailoROIPtr roi, hailo_unique_id_mode_t mode)
     {
         std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_UNIQUE_ID);
         std::vector<HailoUniqueIDPtr> unique_ids;
 
         for (auto obj : objects)
         {
             HailoUniqueIDPtr unique_id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
             // If the unique ID's mode matches the specified mode, add it to the list.
             if(unique_id->get_mode() == mode)
             {
                 unique_ids.emplace_back(unique_id);
             }
         }
         return unique_ids;
     }
 
     // --------------------------------------------------------------------
     // Retrieve HailoUniqueID objects with TRACKING_ID mode.
     inline std::vector<HailoUniqueIDPtr> get_hailo_track_id(HailoROIPtr roi)
     {
         return get_hailo_unique_id_by_mode(roi, TRACKING_ID);
     }
 
     // --------------------------------------------------------------------
     // Retrieve HailoUniqueID objects with GLOBAL_ID mode.
     inline std::vector<HailoUniqueIDPtr> get_hailo_global_id(HailoROIPtr roi)
     {
         return get_hailo_unique_id_by_mode(roi, GLOBAL_ID);
     }
 
     // --------------------------------------------------------------------
     // Retrieve all HailoLandmarks objects from the ROI.
     inline std::vector<HailoLandmarksPtr> get_hailo_landmarks(HailoROIPtr roi)
     {
         std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_LANDMARKS);
         std::vector<HailoLandmarksPtr> landmarks;
 
         for (auto obj : objects)
         {
             landmarks.emplace_back(std::dynamic_pointer_cast<HailoLandmarks>(obj));
         }
         return landmarks;
     }
 
     // --------------------------------------------------------------------
     // Retrieve all HailoROI instances from the ROI's sub-objects.
     inline std::vector<HailoROIPtr> get_hailo_roi_instances(HailoROIPtr roi)
     {
         std::vector<HailoROIPtr> hailo_rois;
 
         // Iterate through all objects attached to the ROI.
         for (auto obj : roi->get_objects())
         {
             // Attempt to cast the object to a HailoROI.
             HailoROIPtr hailo_roi = std::dynamic_pointer_cast<HailoROI>(obj);
             if (hailo_roi)
                 hailo_rois.emplace_back(hailo_roi);
         }
         return hailo_rois;
     }
 
     // --------------------------------------------------------------------
     // Create a flattened bounding box by scaling an input bbox relative to a parent bbox.
     // This adjusts the coordinates and size of the input bbox to the parent's coordinate system.
     inline HailoBBox create_flattened_bbox(const HailoBBox &bbox, const HailoBBox &parent_bbox)
     {
         float xmin = parent_bbox.xmin() + bbox.xmin() * parent_bbox.width();
         float ymin = parent_bbox.ymin() + bbox.ymin() * parent_bbox.height();
 
         float width = bbox.width() * parent_bbox.width();
         float height = bbox.height() * parent_bbox.height();
 
         return HailoBBox(xmin, ymin, width, height);
     }
 
     // --------------------------------------------------------------------
     // Flatten HailoROI sub-objects and move them under a parent ROI.
     // For each sub-object of the specified type, rescale its bounding box to the parent's scale,
     // add it to the parent ROI, and remove it from the original ROI.
     inline void flatten_hailo_roi(HailoROIPtr roi, HailoROIPtr parent_roi, hailo_object_t filter_type)
     {
         std::vector<HailoObjectPtr> objects = roi->get_objects();
         for (uint index = 0; index < objects.size(); index++)
         {
             if (objects[index]->get_type() == filter_type)
             {
                 // Cast the object to a HailoROI.
                 HailoROIPtr sub_obj_roi = std::dynamic_pointer_cast<HailoROI>(objects[index]);
                 // Update its bounding box to be scaled relative to the parent's bbox.
                 sub_obj_roi->set_bbox(std::move(create_flattened_bbox(sub_obj_roi->get_bbox(), roi->get_bbox())));
                 // Add the sub-object to the parent ROI.
                 parent_roi->add_object(sub_obj_roi);
                 // Remove the sub-object from the current ROI.
                 roi->remove_object(index);
                 objects.erase(objects.begin() + index);
                 index--;
             }
         }
     }
 
     // --------------------------------------------------------------------
     // Fixate a point to a new coordinate system defined by a given bbox.
     // The point's coordinates (normalized relative to the ROI) are transformed
     // to be relative to the 'fixate_bbox'.
     inline HailoPoint fixate_point_with_bbox(HailoROIPtr hailo_roi, HailoPoint &point, const HailoBBox &fixate_bbox)
     {
         HailoBBox current_bbox = hailo_roi->get_bbox();
         float xmin = current_bbox.xmin() + point.x() * current_bbox.width();
         float ymin = current_bbox.ymin() + point.y() * current_bbox.height();
 
         return HailoPoint((xmin - fixate_bbox.xmin()) / fixate_bbox.width(), (ymin - fixate_bbox.ymin()) / fixate_bbox.height());
     }
 
     // --------------------------------------------------------------------
     // Fixate (or adjust) the landmarks in an ROI so that they are scaled to a new bbox.
     // For each landmark point in the ROI's landmarks sub-object, compute a new point
     // based on the new bounding box, and update the landmarks with these new coordinates.
     inline void fixate_landmarks_with_bbox(HailoROIPtr hailo_roi, HailoBBox fixate_bbox)
     {
         // Get all landmark sub-objects associated with the ROI.
         std::vector<HailoLandmarksPtr> landmarks_ptrs = get_hailo_landmarks(hailo_roi);
         if (landmarks_ptrs.size() > 0)
         {
             // Use the first landmarks object found.
             HailoLandmarksPtr landmarks_obj = std::dynamic_pointer_cast<HailoLandmarks>(landmarks_ptrs[0]);
             // Retrieve the current landmark points.
             std::vector<HailoPoint> decoded_points = landmarks_obj->get_points();
             std::vector<HailoPoint> new_points;
             new_points.reserve(decoded_points.size());
 
             // For each point, compute its new position relative to the fixate_bbox.
             for (size_t i = 0; i < decoded_points.size(); i++)
             {
                 new_points.emplace_back(fixate_point_with_bbox(hailo_roi, decoded_points[i], fixate_bbox));
             }
             // Update the landmarks with the new, adjusted points.
             landmarks_obj->set_points(new_points);
         }
     }
 }