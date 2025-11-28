use std::fs;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Request sent to training binary
#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub genotype_id: Uuid,
    pub train_config: serde_json::Value,
    pub model_save_path: Option<String>,
}

impl Request {
    /// Parse from JSON string. Do not validate or deserialize the train_config here.
    pub fn from_json_str(json: &str) -> anyhow::Result<Self> {
        let req: Request = serde_json::from_str(json)?;
        Ok(req)
    }

    /// Load from file path. Do not validate or deserialize the train_config here.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

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
