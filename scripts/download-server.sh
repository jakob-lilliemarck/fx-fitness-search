#!/bin/bash
# Download server binary and datasets from GitHub releases
# Usage: ./download-server.sh [version] [destination_dir]
# If no version specified, downloads the latest release

set -e

REPO="jakob-lilliemarck/fx-fitness-search"
DEST_DIR="${2:-.}"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

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

# Download server binary
echo "Downloading server binary..."
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/server"
if curl -fL "$DOWNLOAD_URL" -o "$DEST_DIR/server"; then
    chmod +x "$DEST_DIR/server"
    echo "✓ Downloaded server"
else
    echo "✗ Failed to download server"
    exit 1
fi

# Download and extract datasets
echo "Downloading datasets..."
DATASETS=(
    "beijing-air-quality"
)

for name in "${DATASETS[@]}"; do
    DOWNLOAD_URL="https://github.com/$REPO/releases/download/datasets-$name/$name.tar.gz"
    if curl -fL "$DOWNLOAD_URL" -o "$TEMP_DIR/$name.tar.gz" 2>/dev/null; then
        tar -xzf "$TEMP_DIR/$name.tar.gz" -C "$DEST_DIR"
        echo "✓ Downloaded and extracted $name"
    else
        echo "⚠  Skipped $name (dataset not found)"
    fi
done

echo ""
echo "Server artifacts downloaded to: $DEST_DIR"
echo "Files:"
ls -lh "$DEST_DIR/server" "$DEST_DIR/data" 2>/dev/null || true
