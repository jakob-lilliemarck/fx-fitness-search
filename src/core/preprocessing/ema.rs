use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Ema {
    /// The window size of this ema
    pub(crate) window: usize,
    /// Ema alpha, the "smoothing" factor
    pub(crate) alpha: f32,
    /// The current value
    #[serde(skip, default)]
    pub(crate) value: f32,
    /// Count to track window size saturation
    #[serde(skip, default)]
    pub(crate) count: usize,
}

impl Ema {
    pub fn new(window: usize, alpha: f32) -> Self {
        Self {
            window,
            alpha,
            value: 0.0,
            count: 0,
        }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        if self.count == 0 {
            // Initialize with first value
            self.value = value;
        } else {
            // Apply EMA formula: EMA = value * multiplier + previous_ema * (1 - multiplier)
            self.value = value * self.alpha + self.value * (1.0 - self.alpha);
        }

        self.count += 1;

        // Return None until window is saturated
        if self.count < self.window {
            None
        } else {
            Some(self.value)
        }
    }
}
