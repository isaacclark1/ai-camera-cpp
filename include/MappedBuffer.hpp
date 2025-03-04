// https :  // github.com/erasta/libcamera-opencv/blob/main/main.cpp

#include <errno.h>                            // Defines error codes
#include <stdint.h>                           // Fixed-width integer types
#include <sys/mman.h>                         // Memory-mapping functions
#include <unistd.h>                           // POSIX API

#include <libcamera/libcamera/base/class.h>   // Libcamera class untility macros
#include <libcamera/libcamera/base/flags.h>   // Utilities for handling bitwise flags
#include <libcamera/libcamera/base/span.h>    // Defines Span, a non-owning view over arrays
#include <libcamera/libcamera/framebuffer.h>  // FrameBuffer definitions for libcamera

#include <algorithm>                          // Standard algorithms
#include <map>
#include <vector>

namespace libcamera {
  /**
   * The MappedBuffer class provides an interface for CPU access to memory-mapped buffers.
   * It stores the mapped memory regions ("planes") and handles unmapping during destruction.
   */
  class MappedBuffer {
    private:
      LIBCAMERA_DISABLE_COPY(MappedBuffer) // Disable copy construction during assignment

    public:
      // 'Plane' is a span (view) of uint8_t bytes that represent a memory region
      using Plane = Span<uint8_t>;

      // Destructor: unmaps all mapped memory regions
      ~MappedBuffer();

      // Move constructor and assignment operator to transfer ownership without copying
      MappedBuffer(MappedBuffer &&other) noexcept;              // noexcept = doesn't throw exception
      MappedBuffer &operator=(MappedBuffer &&other) noexcept;

      // Return true if no error occurred during mapping
      bool isValid() const { return error_ == 0; }
      // Return the error code (0 if successful)
      int error() const { return error_; }
      // Returns the vector of mapped memory regions (planes)
      const std::vector<Plane> &planes() const { return planes_; }

    protected:
      // Protected constructor: only derived classes should create instances
      MappedBuffer();

      int error_;                 // Error status (0 if mapping succeeded)
      std::vector<Plane> planes_; // CPU accessible mapped memory regions
      std::vector<Plane> maps_;   // Underlying memory mappings (for unmapping later)
  };

  /**
   * MappedFrameBuffer derives from MappedBuffer and maps a FrameBuffer for CPU access
   */
  class MappedFrameBuffer : public MappedBuffer {
    public:
      // MapFlag specifies the desired protection for the mapping
      enum class MapFlag {
        Read = 1 << 0,            // 1
        Write = 1 << 1,           // 2
        ReadWrite = Read | Write, // 3 - Represents a combination of read and write permissions
      };

      using MapFlags = Flags<MapFlag>;

      // Constructor: Maps the given FrameBuffer with the specified access flags
      MappedFrameBuffer(const FrameBuffer *buffer, MapFlags flags);
  };

  LIBCAMERA_FLAGS_ENABLE_OPERATORS(MappedFrameBuffer::MapFlag)

  // --- Implementation of MappedBuffer ---

  // Default constructor initialises the error status
  MappedBuffer::MappedBuffer() : error_(0) {}

  // Move constructor: Transfers ownership from 'other' without copying
  MappedBuffer::MappedBuffer(MappedBuffer &&other) noexcept {
    *this = std::move(other);
  }

  // Move assignment operator: Transfers memory mappings and invalidates 'other'
  MappedBuffer &MappedBuffer::operator=(MappedBuffer &&other) noexcept {
    error_ = other.error_;
    planes_ = std::move(other.planes_);
    maps_ = std::move(other.maps_);
    other.error_ = -ENOENT;               // Mark moved-from object as invalid
    return *this;
  }

  // Destructor: unmap each mapped region
  MappedBuffer::~MappedBuffer() {
    for (Plane &map : maps_) {
      munmap(map.data(), map.size());
    }
  }

  // --- Implementation of MappedFrameBuffer ---

  // Construct a MappedFrameBuffer by mapping all planes of the provided FrameBuffer
  MappedFrameBuffer::MappedFrameBuffer(const FrameBuffer *buffer, MapFlags flags) {
    // Reserve space for mapped planes
    planes_.reserve(buffer->planes().size());

    int mmapFlags = 0;

    // Set memory protection flags based on the provided MapFlags
    if (flags & MapFlag::Read)
      mmapFlags |= PROT_READ;
    if (flags & MapFlag::Write)
      mmapFlags |= PROT_WRITE;

    // Structure to hold mapping information for a file descriptor
    struct MappedBufferInfo {
      uint8_t *address = nullptr;
      size_t mapLength = 0;
      size_t dmabufLength = 0;
    };

    std::map<int, MappedBufferInfo> mappedBuffers;

    // First pass: determine the required mapping length for each file descriptor
    for (const FrameBuffer::Plane &plane : buffer->planes()) {
      const int fd = plane.fd.get();

      if (mappedBuffers.find(fd) == mappedBuffers.end()) {
        // Get the total length of the buffer associated with this file descriptor
        const size_t length = lseek(fd, 0, SEEK_END);
        mappedBuffers[fd] = MappedBufferInfo{nullptr, 0, length};
      }

      const size_t length = mappedBuffers[fd].dmabufLength;

      // Validate that the plane's offset and length are within the buffer's limits
      if (plane.offset > length || plane.offset + plane.length > length) {
        // Invalidate plane; mapping cannot proceed
        return;
      }

      // Update the mapping length to cover this plane
      size_t &mapLength = mappedBuffers[fd].mapLength;
      mapLength = std::max(mapLength, static_cast<size_t>(plane.offset + plane.length));
    }

    // Second pass: perform the actual mapping and record each plane
    for (const FrameBuffer::Plane &plane : buffer->planes()) {
      const int fd = plane.fd.get();
      auto &info = mappedBuffers[fd];

      if (!info.address) {
        // Map the memory region associated with this file descriptor
        void *address = mmap(nullptr, info.mapLength, mmapFlags, MAP_SHARED, fd, 0);

        if (address == MAP_FAILED) {
          error_ = -errno;
          return;
        }

        info.address = static_cast<uint8_t *>(address);

        // Store the raw mapping for later unmapping
        maps_.emplace_back(info.address, info.mapLength);
      }

      // Record the specific plane by its offset and length within the mapped region
      planes_.emplace_back(info.address + plane.offset, plane.length);
    }
  }
} // namespace libcamera
