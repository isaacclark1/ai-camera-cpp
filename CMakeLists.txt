cmake_minimum_required(VERSION 3.25)

project(AICamera)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -O3 -fconcepts -Wno-ignored-qualifiers -Wno-extra -Wno-stringop-truncation -Wno-reorder)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)
find_package(Threads)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})
message(STATUS "Found HailoRT: " ${HailoRT_INCLUDE_DIRS})

# Look in include folder for header files
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/include/uSockets)
include_directories(/usr/local/include/uWebSockets)
include_directories(/usr/include/libcamera)
include_directories(/usr/include/nlohmann)

# Define source files
file(GLOB SOURCES
    ./src/*.cpp
)

link_libraries(stdc++fs)

# Create executable from the sources 
add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

find_package(ZLIB REQUIRED)

target_link_libraries(${PROJECT_NAME}
  /usr/local/include/uSockets/uSockets.a
  /usr/lib/aarch64-linux-gnu/libcamera.so
  /usr/lib/aarch64-linux-gnu/libcamera-base.so
  ZLIB::ZLIB
)