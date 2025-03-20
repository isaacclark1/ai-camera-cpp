#pragma once
#include <string>

struct HailoDeviceInfo {
  std::string id;
  std::string type;
  std::string architecture;

  HailoDeviceInfo()
    : id("<UNKNOWN>"),
      type("<UNKNOWN>"),
      architecture("<UNKNOWN>") {}

  HailoDeviceInfo(
    std::string device_id,
    std::string device_type,
    std::string device_architecture
  )
    : id(device_id),
      type(device_type),
      architecture(device_architecture) {}
};