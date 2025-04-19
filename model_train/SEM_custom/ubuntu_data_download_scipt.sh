#!/bin/bash

set -e  # Exit immediately on any error

# Install gdown if not available
pip install -q gdown

# File ID and output filename
FILE_ID="11IOlhy9YZ8TlNnnGm28QsCUkxQQkI4JB"
OUTPUT="downloaded_file.zip"
TARGET_DIR="./dataset"

# Download the file
echo "⬇️  Downloading dataset..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT}"

# Create target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Install p7zip (faster and multithreaded unzip tool)
echo "⚙️  Installing 7z (if not already)..."
apt-get update -qq
apt-get install -y -qq p7zip-full

# Unzip with 7z (faster than unzip, supports multithread)
echo "📦 Extracting with 7z..."
7z x "${OUTPUT}" -o"${TARGET_DIR}" -bso0 -bsp1

# Optionally: remove the zip after extraction
rm -f "${OUTPUT}"

echo "✅ Dataset unzipped to ${TARGET_DIR}"
