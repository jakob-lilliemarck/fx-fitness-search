use serde::{Deserialize, Serialize};
use std::fs;

use crate::core::ingestion::Extract;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainConfig {
    pub hidden_size: usize,
    pub sequence_length: usize,
    pub prediction_horizon: usize,
    pub features: Vec<Extract>,
    pub targets: Vec<Extract>,
}

impl TrainConfig {
    pub fn new(
        hidden_size: usize,
        sequence_length: usize,
        prediction_horizon: usize,
        features: Vec<Extract>,
        targets: Vec<Extract>,
    ) -> Self {
        Self {
            hidden_size,
            sequence_length,
            prediction_horizon,
            features,
            targets,
        }
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }
}
