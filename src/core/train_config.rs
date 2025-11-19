use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainConfig {
    pub hidden_size: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
    pub prediction_horizon: usize,
    /// Feature definitions as strings: "temp=TEMP:ZSCORE(48)"
    pub features: Vec<String>,
    /// Target definitions as strings: "target_temp=TEMP"
    pub targets: Vec<String>,
}

impl TrainConfig {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        sequence_length: usize,
        prediction_horizon: usize,
        features: Vec<String>,
        targets: Vec<String>,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
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
