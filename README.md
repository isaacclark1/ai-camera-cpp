# AI Object Detection Server for Raspberry Pi 🎥

## Overview

This project is an AI-powered **object detection application** that utilises a Raspberry Pi 5 and **Hailo AI HAT+** to process video frames in real time.
The app reads frames from a camera, performs detection inference using the Hailo APIs and draws the detections on the video frames.
The server runs a uWebSockets-based WebSocket API to stream the processed frames to clients.

## Features

- 🚀 **Real-time object detection** using the Hailo NPU
- 🎥 **Video frame processing**: drawing detections on video frames and streaming them via a WebSocket server
- 🔔 **Event notifications**: notifications sent to clients such as "person detected"
- 🌡️ **System monitoring**: CPU and RAM statistics streamed to the client.
- ⚡ **Efficient multi-threading**
- 🌍 **REST API** for starting/stopping camera capture and detection inference.

## Technologies Used

- **C++**
- **uWebSockets** (high-performance networking)
- **OpenCV** (image processing)
- **nlohmann/json** (JSON handling)
- **HailoRT** (Hailo AI SDK)
- **Next.js & Shadcn UI components** (for frontend dashboard)

## Installation & Setup

### Prerequisites

- **Raspberry Pi**.
- **Raspberry Pi Camera Module 3**. Other cameras will probably work
- **Raspberry Pi AI HAT+**. Either 13 or 26 TOPS version
- **Hailo SDK** installed. Instructions at: https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html
  - The app uses version 4.20.0 of the Hailo APIs and may not work with future versions.
- **C++** compiler
- **CMake**
- **libcamera**
- **OpenCV**
- **uWebSockets**
- **Node Package Manager (NPM)**

### Build & Run

From the project directory:

```sh
./build.sh
cd build/aarch64
./AICamera <network_file_path> [frame_height, frame_width, jpeg_quality, frame_skip_factor]
```

`network_file_path`: An absolute path to a pre-trained network model to use for object detection REQUIRED. Example: "/home/isaac/projects/ai-camera-cpp/network-files/yolov8m.hef".
`frame_height`: The height of frames to capture from the camera OPTIONAL - default is 1080.
`frame_width`: The width of frames to capture from the camera OPTIONAL - default is 1920.
`jpeg_quality`: The quality to encode the JPEG images sent from the server to clients OPTIONAL. Value should be between 1 and 100.
`frame_skip_factor`: The factor with which to skip frames for inference OPTIONAL. Frames from Raspberry Pi cameras are captured at 30 FPS so a factor of 2 will reduce frame rate to 15 FPS, a factor of 3 will reduce to 10 FPS, etc. This helps to improve performance at the expense of video stream smoothness.

To install network files for the 26 TOPS HAILO-8 device run the `./get_network_files.sh`. For the 13 TOPS HAILO-8L device, models can be found at: https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file.
Additional files for the HAILO-8 device can be found here as well. You MUST use files compiled for the device that you have installed. All files must end in ".hef".

### Running the Next.js Frontend

To run in development mode:

```sh
cd next-app
npm install
npm run dev  # Runs on http://localhost:3000
```

To run in production mode:

```sh
cd next-app
npm install
npm run build
npm run start
```

## API Endpoints

### WebSocket API (`/ws`)

- **`image-stream`** → Binary JPEG frame data
- **`event`** → JSON messages (e.g., `{ "event": "person detected" }`)

### REST API

#### Start Inference

```http
POST /start
```

#### Stop Inference

```http
POST /stop
```

## Monitoring

- CPU & RAM usage is fetched from `/proc/stat` and `/proc/meminfo`
- Hailo device temperature is obtained from `/sys/class/thermal/thermal_zone0/temp`
- CPU stats streamed every **1 second**

## Troubleshooting

### High CPU Usage?

- Increase the **frame_skip_factor** and/or **JPEG quality**

## Next Steps

- Modify and/or extend the Server::handleNewFrame method in `include/server/Server.hpp` to perform custom logic using the detections.

## License

MIT License

## Important Notices

- The Hailo APIs are provided under the LGPL license.
