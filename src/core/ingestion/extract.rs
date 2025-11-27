use crate::core::{interpolation::Interpolation, preprocessing::Node};
use serde::{Deserialize, Serialize};

// Transforms are processed in order
// Values occupy the position at which point they were tranformed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extract {
    pub source: String,
    pub interpolation: Interpolation,
    pub nodes: Vec<Node>,
}

impl Extract {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
            interpolation: Interpolation::Error,
            nodes: Vec::new(),
        }
    }

    pub fn with_node(mut self, n: Node) -> Self {
        self.nodes.push(n);
        self
    }

    pub fn with_interpolation(mut self, i: Interpolation) -> Self {
        self.interpolation = i;
        self
    }
}
