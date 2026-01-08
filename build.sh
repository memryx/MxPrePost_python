#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path to the directory where build.sh is located
PROJECT_ROOT=$(pwd)
NUM_CORES=$(nproc)

echo "Starting build process..."

# 1. build mxapi pymodule from MX_API
echo "\n\n--- Building MX_API ---"
cd "$PROJECT_ROOT/extern/MX_API"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES

# 2. build mx_accl/pymodule
echo "\n\n--- Building mx_accl pymodule ---"
cd "$PROJECT_ROOT/extern/MX_API/mx_accl/pymodule"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES

# 3. build cpp shared library: libmxpipe.so
echo "\n\n--- Building libmxpipe.so ---"
cd "$PROJECT_ROOT"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES

# 4. build pymodule: mxpipe.so
echo "\n\n--- Building mxpipe pymodule ---"
cd "$PROJECT_ROOT/pymodule"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES


# --- Create Symlinks ---
echo "\n\n--- Creating Symbolic Links ---"
# Navigate to the target directory
cd "$PROJECT_ROOT/samples/python"

# 1. Link mxpipe (using wildcard to handle any python version)
# Use 'ln -sf' to overwrite existing links if they exist
ln -sfv "$PROJECT_ROOT/pymodule/build"/mxpipe.cpython-*.so .

# 2. Link mxapi
ln -sfv "$PROJECT_ROOT/extern/MX_API/mx_accl/pymodule/build"/mxapi.cpython-*.so .

echo "Build completed successfully!"