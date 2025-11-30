#!/bin/bash
# Download client binary from GitHub releases
# Usage: ./download-client.sh [version] [destination_dir]
# If no version specified, downloads the latest release

set -e

REPO="jakob-lilliemarck/fx-fitness-search"
DEST_DIR="${2:-.}"

# Get version
if [ -z "$1" ]; then
    echo "No version specified, fetching latest release..."
    VERSION=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep tag_name | cut -d'"' -f4)
else
    VERSION="$1"
fi

if [ -z "$VERSION" ]; then
    echo "Error: Could not determine release version"
    exit 1
fi

echo "Using version: $VERSION"
echo "Destination: $DEST_DIR"

# Download client binary
echo "Downloading client binary..."
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/client"
if curl -fL "$DOWNLOAD_URL" -o "$DEST_DIR/client"; then
    chmod +x "$DEST_DIR/client"
    echo "✓ Downloaded client"
else
    echo "✗ Failed to download client"
    exit 1
fi

echo ""
echo "Client downloaded to: $DEST_DIR/client"
