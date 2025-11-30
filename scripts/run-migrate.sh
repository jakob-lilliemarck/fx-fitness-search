#!/bin/bash
# Download and run the migration binary from GitHub releases
# Usage: ./run-migrate.sh [version]
# If no version specified, downloads the latest release

set -e

REPO="jakob-lilliemarck/fx-fitness-search"
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

# Download binary
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/migrate"
echo "Downloading from: $DOWNLOAD_URL"

if ! curl -fL "$DOWNLOAD_URL" -o "$TEMP_DIR/migrate"; then
    echo "Error: Failed to download migration binary"
    exit 1
fi

chmod +x "$TEMP_DIR/migrate"

# Run migrations
echo "Running migrations..."
export DATABASE_URL="postgres://postgres:${DB_PASSWORD:-postgres}@localhost:5432/fx_durable_ga"
"$TEMP_DIR/migrate"

echo "Migration complete!"
