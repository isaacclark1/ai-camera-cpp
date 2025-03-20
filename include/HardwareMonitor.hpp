#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "global/log_colours.hpp"
#include <optional>
#include <mutex>
#include "global/io_mutex.hpp"

struct CPUStats {
  uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
  double temperature;
};

struct RAMStats {
  uint64_t total, free, used;
};

/**
 * The HardwareMonitor class encapsulates the functionality to monitor hardware stats.
 */
class HardwareMonitor
{
  private:
    std::string cpuLabel;
    CPUStats cpuStats;
    RAMStats ramStats;

    /**
     * Get CPU stats from /proc/stat and update the relevant CPUStats and cpuLabel.
     */
    void setCpuUsage()
    {
      std::ifstream file("/proc/stat");

      if (!file.is_open()) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDYELLOW << "Error: CPU::getCPUStats(): Unable to open /proc/stat" << std::endl << RESET;
        return;
      }

      std::string line;

      if (!std::getline(file, line)) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDYELLOW << "Error: CPU::getCPUStats(): Failed to read /proc/stat" << std::endl << RESET;
        return;
      }

      std::istringstream iss(line);
    
      if (!(
        iss >> this->cpuLabel >> this->cpuStats.user >> this->cpuStats.nice >> this->cpuStats.system >> this->cpuStats.idle
            >> this->cpuStats.iowait >> this->cpuStats.irq >> this->cpuStats.softirq >> this->cpuStats.steal
      )) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDYELLOW << "Error: CPU::getCPUStats(): Failed to parse CPU stats" << std::endl << RESET;
      }
    }

    /**
     * Get CPU tempterature from /sys/class/thermal/thermal_zone0/temp (in celcius) and update CPUStats.
     */
    void setCpuTemperature()
    {
      std::ifstream file("/sys/class/thermal/thermal_zone0/temp");

      // Check if the file was successfully opened
      if (!file.is_open()) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDYELLOW << "Error: CPU::getCPUTemperature(): Unable to open CPU temperature file" << std::endl << RESET;
        return;
      }

      double temp = 0.0;
      file >> temp;

      // Check if the read operation was successful
      if (file.fail()) {
          std::lock_guard<std::mutex> lock(io_mutex);
          std::cerr << BOLDYELLOW << "Error: CPU::getCPUTemperature(): Failed to read temperature" << std::endl << RESET;
          return;
      }

      this->cpuStats.temperature = temp / 1000.0;  // Convert from millidegrees to degrees Celsius
    }

    void setRamStats()
    {
      std::ifstream file("/proc/meminfo");

      if (!file.is_open()) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << BOLDYELLOW << "Error: CPU::getRAMStats(): Unable to open open memory file" << std::endl << RESET;
      }

      std::string key;
      uint64_t value;
      std::string unit;
      
      while (file >> key >> value >> unit) {
             if (key == "MemTotal:")     this->ramStats.total = value;
        else if (key == "MemFree:")      this->ramStats.free = value;
      }

      this->ramStats.used = this->ramStats.total - this->ramStats.free;
    }

  public:
    HardwareMonitor() : cpuLabel(""), cpuStats{0}, ramStats{0}
    {
      this->updateCpuStats();
      this->setRamStats();
    }

    /**
     * Getter for stats.
     * 
     * @return A copy of the CPU stats.
     */
    CPUStats getCpuStats()
    {
      return this->cpuStats;
    }

    /**
     * Update the CPU stats.
     * 
     * @return A copy of the CPU stats.
     */
    CPUStats updateCpuStats()
    {
      this->setCpuUsage();
      this->setCpuTemperature();
      return this->cpuStats;
    }

    /**
     * Getter for RAM stats.
     * 
     * @return A reference to the RAM stats.
     */
    RAMStats& getRamStats()
    {
      this->setRamStats();
      return this->ramStats;
    }

    /**
     * Calculate CPU usage percentage.
     * 
     * @param previous_stats The previous CPU statistics.
     * @param current_stats The current CPU statistics.
     * @return The CPU usage percentage.
     */
    static double calculateCpuUsage(const CPUStats &previous_stats, const CPUStats &current_stats) noexcept
    {
      // Calculate total idle time (idle + iowait)
      const uint64_t previous_stats_idle = previous_stats.idle + previous_stats.iowait;
      const uint64_t current_stats_idle = current_stats.idle + current_stats.iowait;

      // Calculate total non-idle time (user + nice + system + irq + softirq + steal)
      const uint64_t previous_stats_active = previous_stats.user + previous_stats.nice + previous_stats.system +
                                          previous_stats.irq + previous_stats.softirq + previous_stats.steal;
      const uint64_t current_stats_active = current_stats.user + current_stats.nice + current_stats.system +
                                            current_stats.irq + current_stats.softirq + current_stats.steal;

      // Calculate total CPU time (idle + non-idle)
      const uint64_t previous_stats_total = previous_stats_idle + previous_stats_active;
      const uint64_t current_stats_total = current_stats_idle + current_stats_active;

      // Compute differences between the previous and current values
      const uint64_t total_diff = current_stats_total - previous_stats_total;
      const uint64_t idle_diff = current_stats_idle - previous_stats_idle;

      // Prevent division by zero
      if (total_diff == 0) return 0.0;

      // Compute CPU usage percentage: 100 * (total - idle) / total
      return (100.0 * (total_diff - idle_diff)) / total_diff;
    }
};