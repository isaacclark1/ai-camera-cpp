#pragma once

#include <mutex>

static std::mutex io_mutex; // Thread safe I/O operations