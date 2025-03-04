#!/bin/bash
# This script builds a project using CMake for specific architectures.

# Set the default C compiler to gcc.
export CC=gcc 
# Set the default C++ compiler to g++.
export CXX=g++

# Declare an associative array (dictionary) named COMPILER.
# This array maps architecture names to the corresponding GCC executable paths.
declare -A COMPILER=( 
    [x86_64]=/usr/bin/gcc
    [aarch64]=/usr/bin/aarch64-linux-gnu-gcc
    [armv7l]=/usr/bin/arm-linux-gnueabi-gcc 
)

# Iterate over the desired architectures.
# In this case, the loop runs for the "aarch64" architecture.
for ARCH in aarch64
do
    # Print a message indicating which architecture is being built.
    echo "-I- Building ${ARCH}"
    
    # Create a build directory for the current architecture (if it doesn't already exist).
    mkdir -p build/${ARCH}
    
    # Run CMake to configure the project:
    # -H. tells CMake to use the current directory as the source directory.
    # -Bbuild/${ARCH} tells CMake to place the build files in the build/<ARCH> directory.
    cmake -H. -Bbuild/${ARCH}
    
    # Build the project using the generated build files in the build/<ARCH> directory.
    cmake --build build/${ARCH}
done

# After the build process, check if the file "hailort.log" exists.
if [[ -f "hailort.log" ]]; then
    # If the log file exists, remove it.
    rm hailort.log
fi