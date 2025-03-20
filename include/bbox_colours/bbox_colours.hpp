#pragma once
#include <unordered_map>
#include <opencv2/opencv.hpp>

std::unordered_map<std::string, cv::Scalar> bbox_colours = {
  {"person", cv::Scalar(50, 200, 0)},          // Green
  {"bicycle", cv::Scalar(0, 0, 200)},         // Red
  {"car", cv::Scalar(200, 50, 0)},           // Blue
  {"motorcycle", cv::Scalar(180, 180, 0)},   // Teal
  {"airplane", cv::Scalar(0, 255, 255)},     // Cyan
  {"bus", cv::Scalar(200, 0, 140)},         // Purple
  {"train", cv::Scalar(255, 0, 255)},       // Magenta
  {"truck", cv::Scalar(255, 140, 0)},       // Orange
  {"boat", cv::Scalar(128, 0, 128)},        // Dark Purple
  {"traffic light", cv::Scalar(0, 255, 0)}, // Bright Green
  {"fire hydrant", cv::Scalar(255, 0, 0)},  // Bright Red
  {"stop sign", cv::Scalar(255, 0, 0)},     // Bright Red (same as fire hydrant)
  {"parking meter", cv::Scalar(128, 128, 128)}, // Gray
  {"bench", cv::Scalar(139, 69, 19)},       // Brown
  {"bird", cv::Scalar(0, 191, 255)},        // Deep Sky Blue
  {"cat", cv::Scalar(255, 20, 147)},        // Deep Pink
  {"dog", cv::Scalar(0, 180, 240)},         // Yellow
  {"horse", cv::Scalar(75, 0, 130)},        // Indigo
  {"sheep", cv::Scalar(0, 128, 0)},         // Dark Green
  {"cow", cv::Scalar(165, 42, 42)},         // Brown
  {"elephant", cv::Scalar(128, 128, 0)},    // Olive
  {"bear", cv::Scalar(139, 0, 0)},          // Dark Red
  {"zebra", cv::Scalar(255, 255, 255)},     // White
  {"giraffe", cv::Scalar(218, 165, 32)},    // Goldenrod
  {"backpack", cv::Scalar(255, 215, 0)},    // Gold
  {"umbrella", cv::Scalar(70, 130, 180)},   // Steel Blue
  {"handbag", cv::Scalar(176, 224, 230)},   // Powder Blue
  {"tie", cv::Scalar(240, 128, 128)},       // Light Coral
  {"suitcase", cv::Scalar(47, 79, 79)},     // Dark Slate Gray
  {"frisbee", cv::Scalar(255, 69, 0)},      // Orange Red
  {"skis", cv::Scalar(100, 149, 237)},      // Cornflower Blue
  {"snowboard", cv::Scalar(70, 130, 180)},  // Steel Blue
  {"sports ball", cv::Scalar(255, 165, 0)}, // Orange
  {"kite", cv::Scalar(0, 191, 255)},        // Deep Sky Blue
  {"baseball bat", cv::Scalar(139, 69, 19)},// Saddle Brown
  {"baseball glove", cv::Scalar(210, 180, 140)}, // Tan
  {"skateboard", cv::Scalar(255, 140, 0)},  // Dark Orange
  {"surfboard", cv::Scalar(0, 255, 127)},   // Spring Green
  {"tennis racket", cv::Scalar(0, 206, 209)}, // Dark Turquoise
  {"bottle", cv::Scalar(255, 20, 147)},     // Deep Pink
  {"wine glass", cv::Scalar(255, 182, 193)},// Light Pink
  {"cup", cv::Scalar(255, 228, 196)},       // Bisque
  {"fork", cv::Scalar(105, 105, 105)},      // Dim Gray
  {"knife", cv::Scalar(169, 169, 169)},     // Dark Gray
  {"spoon", cv::Scalar(192, 192, 192)},     // Silver
  {"bowl", cv::Scalar(255, 250, 205)},      // Lemon Chiffon
  {"banana", cv::Scalar(255, 255, 0)},      // Yellow
  {"apple", cv::Scalar(220, 20, 60)},       // Crimson
  {"sandwich", cv::Scalar(255, 215, 0)},    // Gold
  {"orange", cv::Scalar(255, 165, 0)},      // Orange
  {"broccoli", cv::Scalar(0, 128, 0)},      // Dark Green
  {"carrot", cv::Scalar(255, 140, 0)},      // Dark Orange
  {"hot dog", cv::Scalar(255, 99, 71)},     // Tomato
  {"pizza", cv::Scalar(255, 69, 0)},        // Orange Red
  {"donut", cv::Scalar(255, 192, 203)},     // Pink
  {"cake", cv::Scalar(255, 222, 173)},      // Navajo White
  {"chair", cv::Scalar(139, 69, 19)},       // Saddle Brown
  {"couch", cv::Scalar(160, 82, 45)},       // Sienna
  {"potted plant", cv::Scalar(34, 139, 34)},// Forest Green
  {"bed", cv::Scalar(205, 92, 92)},         // Indian Red
  {"dining table", cv::Scalar(139, 0, 139)},// Dark Magenta
  {"toilet", cv::Scalar(255, 239, 213)},    // Papaya Whip
  {"tv", cv::Scalar(30, 144, 255)},         // Dodger Blue
  {"laptop", cv::Scalar(0, 0, 128)},        // Navy Blue
  {"mouse", cv::Scalar(112, 128, 144)},     // Slate Gray
  {"remote", cv::Scalar(255, 250, 205)},    // Lemon Chiffon
  {"keyboard", cv::Scalar(176, 196, 222)},  // Light Steel Blue
  {"cellphone", cv::Scalar(105, 105, 105)}, // Dim Gray
  {"microwave", cv::Scalar(218, 165, 32)},  // Goldenrod
  {"oven", cv::Scalar(184, 134, 11)},       // Dark Goldenrod
  {"toaster", cv::Scalar(205, 133, 63)},    // Peru
  {"sink", cv::Scalar(240, 255, 255)},      // Azure
  {"refrigerator", cv::Scalar(70, 130, 180)},// Steel Blue
  {"book", cv::Scalar(255, 215, 0)},        // Gold
  {"clock", cv::Scalar(255, 69, 0)},        // Orange Red
  {"vase", cv::Scalar(255, 160, 122)},      // Light Salmon
  {"scissors", cv::Scalar(255, 99, 71)},    // Tomato
  {"teddy bear", cv::Scalar(255, 182, 193)},// Light Pink
  {"hair drier", cv::Scalar(255, 228, 196)},// Bisque
  {"tooth brush", cv::Scalar(255, 245, 238)}// Seashell
};