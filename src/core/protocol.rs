use std::fs;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Request sent to training binary via subprocess communication
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

/// Common request variables passed to training subprocess
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RequestVariables {
    pub prediction_horizon: usize,
    pub epochs: usize,
    pub patience: usize,
    pub validation_start_epoch: usize,
    pub batch_size: usize,
}

impl RequestVariables {
    pub fn new(prediction_horizon: usize) -> Self {
        // Provide sensible defaults; CLI can override
        Self {
            prediction_horizon,
            epochs: 25,
            patience: 5,
            validation_start_epoch: 10,
            batch_size: 100,
        }
    }

    /// Create RequestVariables with all parameters explicitly set
    pub fn with_training_params(
        prediction_horizon: usize,
        epochs: usize,
        patience: usize,
        validation_start_epoch: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            prediction_horizon,
            epochs,
            patience,
            validation_start_epoch,
            batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_from_json_str() {
        let json = r#"{"genotype_id":"550e8400-e29b-41d4-a716-446655440000","train_config":{},"model_save_path":null}"#;
        let req = Request::from_json_str(json).unwrap();
        assert_eq!(req.model_save_path, None);
    }

    #[test]
    fn test_request_serialization() {
        let original = Request {
            genotype_id: Uuid::nil(),
            train_config: serde_json::json!({"hidden_size": 128}),
            model_save_path: Some("./model".to_string()),
        };

        let json = serde_json::to_string(&original).unwrap();
        let loaded: Request = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.genotype_id, original.genotype_id);
        assert_eq!(loaded.model_save_path, original.model_save_path);
    }

    #[test]
    fn test_response_roundtrip() {
        let response = Response {
            fitness: 0.42,
            genotype_id: Uuid::nil(),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: Response = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.fitness, response.fitness);
        assert_eq!(deserialized.genotype_id, response.genotype_id);
    }

    #[test]
    fn test_request_variables_new() {
        let vars = RequestVariables::new(10);
        assert_eq!(vars.prediction_horizon, 10);
    }
}
