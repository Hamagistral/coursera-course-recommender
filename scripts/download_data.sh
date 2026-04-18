#!/usr/bin/env bash
# Download the Coursera dataset from Kaggle
# Prerequisites: kaggle CLI configured with API key
# Usage: bash scripts/download_data.sh

set -euo pipefail

DATASET="elvinrustam/coursera-dataset"
TARGET_FILE="CourseraDataset-Unclean.csv"
DATA_DIR="data/raw"

echo "==> Downloading Coursera dataset from Kaggle..."
mkdir -p "$DATA_DIR"

# Download and unzip into data/raw
kaggle datasets download -d "$DATASET" -p "$DATA_DIR" --unzip

# The downloaded file may have a different name; find it
DOWNLOADED=$(find "$DATA_DIR" -name "*.csv" | head -1)

if [ -z "$DOWNLOADED" ]; then
    echo "ERROR: No CSV file found in $DATA_DIR after download."
    exit 1
fi

# Rename to expected filename if needed
if [ "$DOWNLOADED" != "$DATA_DIR/$TARGET_FILE" ]; then
    mv "$DOWNLOADED" "$DATA_DIR/$TARGET_FILE"
    echo "==> Renamed to $DATA_DIR/$TARGET_FILE"
fi

echo "✅ Dataset downloaded: $DATA_DIR/$TARGET_FILE"
wc -l "$DATA_DIR/$TARGET_FILE"
