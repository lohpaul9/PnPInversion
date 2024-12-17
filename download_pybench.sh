#!/bin/bash

# Variables
FILE_ID="1qNNAbkd90qGTgwKdMppMpMU8Cegbj8bF"
DEST_FILE="downloaded_file.zip"
DATA_FOLDER="data"

# Check if gdown is installed, install it if not
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing gdown..."
    pip install gdown --user
fi

# Create data folder if it doesn't exist
mkdir -p "$DATA_FOLDER"

# Download the file using gdown
echo "Downloading file using gdown..."
gdown --id "$FILE_ID" -O "$DEST_FILE"

# Check if download was successful
if [ -f "$DEST_FILE" ]; then
    echo "Download completed successfully."
else
    echo "Error: File download failed."
    exit 1
fi

# Unzip the file into the data folder
echo "Unzipping file into '$DATA_FOLDER'..."
unzip -o "$DEST_FILE" -d "$DATA_FOLDER"

# Verify unzip success
if [ $? -eq 0 ]; then
    echo "File unzipped successfully into '$DATA_FOLDER'."
else
    echo "Error: Unzipping failed."
    exit 1
fi

# Clean up downloaded zip file
rm "$DEST_FILE"
echo "Temporary file removed."

echo "Done!"