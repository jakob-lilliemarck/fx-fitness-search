use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Sin {
    period: f32,
}

impl Sin {
    pub fn new(period: f32) -> Self {
        Self { period }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        Some((2.0 * PI * value / self.period).sin())
    }
}
