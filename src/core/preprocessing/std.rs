use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Std {
    pub window: usize,
    #[serde(skip, default)]
    buffer: VecDeque<f32>,
}

impl Std {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: window_size,
            buffer: VecDeque::with_capacity(window_size),
        }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        self.buffer.push_back(value);

        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }

        if self.buffer.len() == self.window {
            let mean: f32 = self.buffer.iter().sum::<f32>() / self.window as f32;
            let variance: f32 =
                self.buffer.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.window as f32;
            Some(variance.sqrt())
        } else {
            None
        }
    }
}
