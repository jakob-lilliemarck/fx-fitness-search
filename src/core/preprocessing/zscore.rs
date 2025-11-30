use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZScore {
    pub(crate) window: usize,
    #[serde(skip, default)]
    pub(crate) values: VecDeque<f32>,
    #[serde(skip, default)]
    pub(crate) count: usize,
}

impl ZScore {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            values: VecDeque::with_capacity(window),
            count: 0,
        }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        self.values.push_back(value);
        if self.values.len() > self.window {
            self.values.pop_front();
        }
        self.count += 1;

        // If the window is not saturated, return None
        if self.count < self.window {
            return None;
        }

        // Calculate the SMA
        let mean = self.values.iter().sum::<f32>() / self.window as f32;

        // Calculate variance
        let variance =
            self.values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / self.values.len() as f32;

        // Calculate the standard deviation
        let std = variance.sqrt();

        // Handle division by zero: if std=0, all values are identical
        // In this case, zscore should be 0 (no deviation from mean)
        if std == 0.0 || std.is_nan() {
            Some(0.0)
        } else {
            Some((value - mean) / std)
        }
    }
}
