use std::fs;

use crate::core::train_config::TrainConfig;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Request sent to training binary
#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub genotype_id: Uuid,
    pub train_config: TrainConfig,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub model_save_path: Option<String>,
}

impl Request {
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

/// Response from training binary
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub fitness: f64,
    pub genotype_id: Uuid,
}

/// Variables stored and deserialized from the request record
#[derive(Debug, Serialize, Deserialize)]
pub struct RequestVariables {
    pub prediction_horizon: usize,
}

impl RequestVariables {
    pub fn new(prediction_horizon: usize) -> Self {
        Self { prediction_horizon }
    }
}
