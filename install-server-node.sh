#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BINARY_URL="https://github.com/jakob-lilliemarck/fx-fitness-search/releases/download/v0.0.1/server"
INSTALL_DIR="${INSTALL_DIR:-/opt/fx-server}"
SERVICE_NAME="fx-server"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root for systemd installation
check_root_for_systemd() {
    if [[ "$INSTALL_SYSTEMD" == "true" && "$EUID" -ne 0 ]]; then
        log_error "SystemD installation requires root privileges"
        exit 1
    fi
}

# Download the binary
download_binary() {
    log_info "Downloading server binary from $BINARY_URL..."
    
    if ! curl -fsSL -o "$INSTALL_DIR/server" "$BINARY_URL"; then
        log_error "Failed to download binary"
        exit 1
    fi
    
    chmod +x "$INSTALL_DIR/server"
    log_info "Binary downloaded to $INSTALL_DIR/server"
}

# Prompt for environment variables
configure_environment() {
    local env_file="$INSTALL_DIR/.env.server"
    
    log_info "Configuring environment variables..."
    
    # HOST_ID
    read -p "Enter HOST_ID (unique UUID for this node): " host_id
    if [[ -z "$host_id" ]]; then
        log_error "HOST_ID is required"
        exit 1
    fi
    
    # DATABASE_URL
    read -p "Enter DATABASE_URL: " database_url
    if [[ -z "$database_url" ]]; then
        log_error "DATABASE_URL is required"
        exit 1
    fi
    
    # LEASE_SECONDS
    read -p "Enter LEASE_SECONDS (default: 300): " lease_seconds
    lease_seconds=${lease_seconds:-300}
    
    # SHUTDOWN_TIMEOUT_SECONDS
    read -p "Enter SHUTDOWN_TIMEOUT_SECONDS (default: 30): " shutdown_timeout
    shutdown_timeout=${shutdown_timeout:-30}
    
    # MAX_WORKERS
    read -p "Enter MAX_WORKERS (default: number of CPU cores): " max_workers
    if [[ -z "$max_workers" ]]; then
        max_workers=$(nproc 2>/dev/null || echo "4")
    fi
    
    # MODEL_SAVE_PATH
    read -p "Enter MODEL_SAVE_PATH (default: $INSTALL_DIR/models): " model_save_path
    model_save_path=${model_save_path:-$INSTALL_DIR/models}
    
    # Create the .env file
    cat > "$env_file" << EOF
HOST_ID=$host_id
DATABASE_URL=$database_url
LEASE_SECONDS=$lease_seconds
SHUTDOWN_TIMEOUT_SECONDS=$shutdown_timeout
MAX_WORKERS=$max_workers
MODEL_SAVE_PATH=$model_save_path
EOF
    
    chmod 600 "$env_file"
    log_info "Environment configuration saved to $env_file"
}

# Create necessary directories
create_directories() {
    mkdir -p "$INSTALL_DIR/models"
    log_info "Created directories in $INSTALL_DIR"
}

# Create systemd service file
create_systemd_service() {
    local service_file="/etc/systemd/system/${SERVICE_NAME}.service"
    local install_dir_abs=$(cd "$INSTALL_DIR" && pwd)
    
    log_info "Creating systemd service file..."
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=FX Durable GA Server Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$install_dir_abs
EnvironmentFile=$install_dir_abs/.env.server
ExecStart=$install_dir_abs/server
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    sudo systemctl start $SERVICE_NAME
    
    log_info "SystemD service installed and enabled"
    log_info "Server is now running and will start on boot"
}

# Print usage
print_usage() {
    cat << EOF
FX Durable GA Server Node Installer

Usage: ./install-server-node.sh [OPTIONS]

Options:
  -d, --dir DIR           Installation directory (default: /opt/fx-server)
  -s, --systemd           Install systemd service (requires root)
  -h, --help              Show this help message

Examples:
  # Install in default location (/opt/fx-server)
  sudo ./install-server-node.sh

  # Install in custom directory with systemd service
  sudo ./install-server-node.sh -d /custom/path -s

  # Download and run in one command
  bash <(curl -fsSL https://raw.githubusercontent.com/jakob-lilliemarck/fx-fitness-search/main/install-server-node.sh) -d /opt/fx-server
EOF
}

# Main
main() {
    local install_systemd=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            -s|--systemd)
                install_systemd=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    INSTALL_SYSTEMD="$install_systemd"
    
    # Create install directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
    
    check_root_for_systemd
    download_binary
    create_directories
    configure_environment
    
    echo ""
    log_info "Installation complete!"
    log_info "Configuration file: $INSTALL_DIR/.env.server"
    echo ""
    
    if [[ "$install_systemd" == "true" ]]; then
        create_systemd_service
    else
        echo "To start the server, run:"
        echo "  cd $INSTALL_DIR"
        echo "  ./server"
    fi
}

main "$@"
