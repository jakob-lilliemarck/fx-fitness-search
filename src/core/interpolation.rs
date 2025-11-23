use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::core::ingestion;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Missing required values")]
    Required,
}

impl From<Error> for ingestion::Error {
    fn from(value: Error) -> Self {
        ingestion::Error::InterpolationError(Box::new(value))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Interpolation {
    Error,
    Linear(LinearInterpolator),
}

impl Interpolation {
    pub fn interpolate(&mut self, input: Option<f32>) -> Result<Option<f32>, Error> {
        match self {
            Self::Error => {
                // Error strategy: return the value if present, fail only if missing
                if input.is_some() {
                    Ok(input)
                } else {
                    Err(Error::Required)
                }
            }
            Self::Linear(i) => Ok(i.interpolate(input)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearInterpolator {
    pub window_size: usize,
    #[serde(skip, default)]
    prev_values: VecDeque<f32>,
}

impl LinearInterpolator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            prev_values: VecDeque::with_capacity(window_size),
        }
    }

    pub fn interpolate(&mut self, input: Option<f32>) -> Option<f32> {
        if let Some(value) = input {
            self.prev_values.push_back(value);
            if self.prev_values.len() > self.window_size {
                self.prev_values.pop_front();
            }
            return Some(value);
        }

        // Not enough history
        if self.prev_values.len() < 2 {
            return None;
        }

        // Fit line: y = mx + b
        let n = self.prev_values.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = self.prev_values.iter().sum::<f32>() / n;

        let slope = self
            .prev_values
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
            .sum::<f32>()
            / self
                .prev_values
                .iter()
                .enumerate()
                .map(|(i, _)| (i as f32 - x_mean).powi(2))
                .sum::<f32>();

        let intercept = y_mean - slope * x_mean;
        let next_x = n;

        Some(slope * next_x + intercept)
    }
}
