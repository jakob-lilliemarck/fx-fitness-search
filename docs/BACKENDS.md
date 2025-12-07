# Backend Selection Guide

This project supports multiple Burn backends for training, selectable at compile time via feature flags.

## Available Backends

### WGPU (Default) - GPU Accelerated
- **Feature:** `backend-wgpu` (enabled by default)
- **Hardware:** NVIDIA, AMD, Intel GPUs via Vulkan/Metal/DirectX
- **Use case:** GPU training on machines with discrete or integrated GPUs
- **Dependencies:** Only requires GPU drivers (no CUDA Toolkit needed)

### NdArray - CPU Only
- **Feature:** `backend-ndarray`
- **Hardware:** CPU only
- **Use case:** Testing, CI/CD, machines without GPU
- **Dependencies:** None

## Building with Different Backends

### Build with WGPU (default)
```bash
cargo build --release --bin beijing
```

Or explicitly:
```bash
cargo build --release --bin beijing --features backend-wgpu
```

### Build with NdArray
```bash
cargo build --release --bin beijing --no-default-features --features backend-ndarray
```

### Running Tests

**NdArray backend tests (fast, CPU-only):**
```bash
SQLX_OFFLINE=true cargo test --no-default-features --features backend-ndarray
```
Runs 93 generic tests + 1 doctest = **94 tests**

**WGPU backend tests (includes GPU integration tests):**
```bash
SQLX_OFFLINE=true cargo test --features backend-wgpu
```
Runs 93 generic tests + 4 WGPU-specific tests + 1 doctest = **98 tests**

The WGPU-specific tests verify:
- Device creation (GPU or CPU fallback)
- Model forward pass on WGPU
- Training epoch with WGPU
- Validation epoch with WGPU

## Hardware Requirements

### For WGPU Backend on Linux (EndeavourOS)

**NVIDIA GTX 960 (Your setup):**
- Vulkan support: ✅ Yes (via Mesa or proprietary NVIDIA drivers)
- Installation: `sudo pacman -S nvidia nvidia-utils vulkan-icd-loader`
- Verification: `vulkaninfo` (install via `sudo pacman -S vulkan-tools`)

**General Requirements:**
- GPU with Vulkan, Metal, or DirectX support
- Up-to-date GPU drivers

### For NdArray Backend
- No special requirements
- Works on any CPU

## Checking Your GPU Setup

On Linux, check if Vulkan is available:
```bash
# Check for Vulkan support
vulkaninfo | head -n 20

# Or check GPU devices
lspci | grep -E "VGA|3D"

# For NVIDIA specifically
nvidia-smi  # If using proprietary drivers
```

## Performance Characteristics

| Backend | Speed (GTX 960) | Binary Size | Dependencies |
|---------|----------------|-------------|--------------|
| WGPU    | ~10-50x faster | ~30MB       | GPU drivers  |
| NdArray | Baseline       | ~20MB       | None         |

## Troubleshooting

### WGPU: "No adapter found"
- Ensure GPU drivers are installed
- Check Vulkan support: `vulkaninfo`
- Try fallback to CPU: WGPU will automatically use CPU if GPU unavailable

### WGPU: Compilation recursion limit error
The `#![recursion_limit = "256"]` attribute is already set in affected binaries.

### Build errors about SQLx
Use offline mode during compilation:
```bash
SQLX_OFFLINE=true cargo build
```

## Implementation Details

Backend selection is implemented in `src/backend.rs` with feature-gated type aliases:
- `Backend`: The autodiff-enabled backend type
- `Device`: The device type for the selected backend
- `default_device()`: Get the default device

All training code is generic over the `Backend` trait, so no code changes are needed when switching backends.
