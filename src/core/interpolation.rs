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

        // If we have exactly one value, return it (use previous value for gap)
        if self.prev_values.len() == 1 {
            return self.prev_values.back().copied();
        }

        // Need at least 2 points to fit a line
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_size_1_returns_previous_value_on_gap() {
        // Behavior: with window_size = 1, if we have exactly one prior value and encounter a gap (None),
        // we return the previous value to avoid breaking ingestion with None values.
        let mut interpolator = LinearInterpolator::new(1);

        // First observed value is passed through
        assert_eq!(interpolator.interpolate(Some(5.0)), Some(5.0));
        // Gap -> use the last seen value
        assert_eq!(interpolator.interpolate(None), Some(5.0));
        // Another gap -> still use the last seen value
        assert_eq!(interpolator.interpolate(None), Some(5.0));
    }

    #[test]
    fn window_size_2_interpolates_linearly_on_gap() {
        // Behavior: with window_size >= 2, we have enough history to fit a line and extrapolate the next value.
        // Example sequence: 10.0, 20.0, None -> expected 30.0
        let mut interpolator = LinearInterpolator::new(2);

        assert_eq!(interpolator.interpolate(Some(10.0)), Some(10.0));
        assert_eq!(interpolator.interpolate(Some(20.0)), Some(20.0));
        // Gap -> linear extrapolation from [10.0, 20.0] should produce 30.0
        assert_eq!(interpolator.interpolate(None), Some(30.0));
    }

    #[test]
    fn no_history_gap_returns_none() {
        // Behavior: if we encounter a gap before any values have been observed, return None.
        // This is expected at the start of ingestion when the interpolator is empty.
        let mut interpolator = LinearInterpolator::new(2);

        assert_eq!(interpolator.interpolate(None), None);
        assert_eq!(interpolator.interpolate(None), None);
    }

    #[test]
    fn window_size_respects_maximum() {
        // Behavior: the window should only keep the most recent window_size values.
        // With window_size=2, adding a third value should drop the first.
        let mut interpolator = LinearInterpolator::new(2);

        assert_eq!(interpolator.interpolate(Some(10.0)), Some(10.0));
        assert_eq!(interpolator.interpolate(Some(20.0)), Some(20.0));
        assert_eq!(interpolator.interpolate(Some(30.0)), Some(30.0));
        // Now a gap should interpolate using [20.0, 30.0], not [10.0, 20.0]
        // Linear regression of [20.0, 30.0] should produce 40.0
        assert_eq!(interpolator.interpolate(None), Some(40.0));
    }

    #[test]
    fn larger_window_size_with_multiple_gaps() {
        // Behavior: with window_size=3, we can handle gaps and continue interpolating
        // based on the most recent history.
        let mut interpolator = LinearInterpolator::new(3);

        assert_eq!(interpolator.interpolate(Some(5.0)), Some(5.0));
        assert_eq!(interpolator.interpolate(Some(10.0)), Some(10.0));
        assert_eq!(interpolator.interpolate(Some(15.0)), Some(15.0));
        // Gap -> linear extrapolation from [5.0, 10.0, 15.0] should produce 20.0
        assert_eq!(interpolator.interpolate(None), Some(20.0));
        // Another gap -> still use previous value (only 1 in history now)
        assert_eq!(interpolator.interpolate(None), Some(20.0));
        // New value resets history
        assert_eq!(interpolator.interpolate(Some(25.0)), Some(25.0));
    }

    #[test]
    fn error_strategy_fails_on_missing() {
        // Behavior: Interpolation::Error strategy fails when encountering a gap.
        let mut interpolation = Interpolation::Error;

        assert!(interpolation.interpolate(Some(5.0)).is_ok());
        assert!(interpolation.interpolate(None).is_err());
    }

    #[test]
    fn error_strategy_passes_through_values() {
        // Behavior: Interpolation::Error strategy returns values when present.
        let mut interpolation = Interpolation::Error;

        let result = interpolation.interpolate(Some(42.0));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(42.0));
    }
}
