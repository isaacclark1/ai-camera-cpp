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
    const std::string device_id,
    const std::string device_type,
    const std::string device_architecture
  )
    : id(device_id),
      type(device_type),
      architecture(device_architecture) {}
};