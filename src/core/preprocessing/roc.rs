use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Roc {
    pub offset: usize,
    #[serde(skip, default)]
    buffer: VecDeque<f32>,
}

impl Roc {
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            buffer: VecDeque::with_capacity(offset + 1),
        }
    }

    pub fn process(&mut self, input: f32) -> Option<f32> {
        self.buffer.push_back(input);

        // Keep only the values we need
        if self.buffer.len() > self.offset + 1 {
            self.buffer.pop_front();
        }

        // Need at least offset+1 values to compute ROC
        if self.buffer.len() > self.offset {
            Some((input - self.buffer[0]) / self.offset as f32)
        } else {
            None
        }
    }
}
