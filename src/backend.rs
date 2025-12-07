//! Backend configuration for training.
//!
//! This module provides type aliases for the backend and device based on compile-time
//! feature flags. By default, the WGPU backend is used for GPU acceleration.
//!
//! # Feature Flags
//!
//! - `backend-wgpu` (default): Use WGPU backend for GPU acceleration via Vulkan/Metal/DirectX
//! - `backend-ndarray`: Use NdArray backend for CPU-only training
//!
//! # Example
//!
//! ```rust,no_run
//! use fx_durable_ga_app::backend::{Backend, default_device};
//!
//! let device = default_device();
//! // let model = MyModel::<Backend>::new(&device);
//! ```

#[cfg(feature = "backend-wgpu")]
pub use burn_wgpu::Wgpu as InnerBackend;

#[cfg(feature = "backend-ndarray")]
pub use burn_ndarray::NdArray as InnerBackend;

/// The backend type to use for training, with autodiff support.
pub type Backend = burn::backend::Autodiff<InnerBackend>;

/// The device type corresponding to the selected backend.
#[cfg(feature = "backend-wgpu")]
pub type Device = burn_wgpu::WgpuDevice;

#[cfg(feature = "backend-ndarray")]
pub type Device = burn_ndarray::NdArrayDevice;

/// Get the default device for the selected backend.
pub fn default_device() -> Device {
    #[cfg(feature = "backend-wgpu")]
    tracing::info!("Using WGPU backend");
    
    #[cfg(feature = "backend-ndarray")]
    tracing::info!("Using NdArray (CPU) backend");
    
    let device = Device::default();
    tracing::info!("Device initialized: {:?}", device);
    device
}
