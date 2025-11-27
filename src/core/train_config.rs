use serde::{Deserialize, Serialize};
use std::fs;

use crate::core::ingestion::Extract;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub hidden_size: usize,
    pub sequence_length: usize,
    pub prediction_horizon: usize,
    pub features: Vec<Extract>,
    pub targets: Vec<Extract>,
    /// Weight decay for L2 regularization. None = disabled
    pub weight_decay: Option<f32>,
    /// Gradient clipping norm. None = disabled
    pub grad_clip: Option<f32>,
    /// Early stopping patience: number of epochs without improvement before stopping. None = disabled
    pub patience: Option<usize>,
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
            weight_decay: None,
            grad_clip: None,
            patience: None,
        }
    }

    /// Builder method to set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }

    /// Builder method to set gradient clipping
    pub fn with_grad_clip(mut self, grad_clip: f32) -> Self {
        self.grad_clip = Some(grad_clip);
        self
    }

    /// Builder method to set early stopping patience
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = Some(patience);
        self
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TrainConfig {
        TrainConfig::new(
            64,
            10,
            1,
            vec![Extract::new("feature1"), Extract::new("feature2")],
            vec![Extract::new("target1")],
        )
    }

    fn create_temp_path() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!(
            "/tmp/train_config_test_{}_{}.json",
            std::process::id(),
            timestamp
        )
    }

    #[test]
    fn test_new_creates_config_with_correct_values() {
        let config = TrainConfig::new(
            32,
            20,
            2,
            vec![Extract::new("f1")],
            vec![Extract::new("t1")],
        );

        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.sequence_length, 20);
        assert_eq!(config.prediction_horizon, 2);
        assert_eq!(config.features.len(), 1);
        assert_eq!(config.targets.len(), 1);
    }

    #[test]
    fn test_save_creates_valid_json_file() {
        let config = create_test_config();
        let path = create_temp_path();

        config.save(&path).expect("Failed to save config");

        // Verify file exists and can be parsed as valid JSON
        let contents = fs::read_to_string(&path).expect("Failed to read file");
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&contents);
        assert!(parsed.is_ok(), "File does not contain valid JSON");

        // Cleanup - remove before exiting
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_roundtrip_save_load_preserves_all_fields() {
        let config = TrainConfig::new(
            128,
            50,
            5,
            vec![
                Extract::new("temp"),
                Extract::new("humidity"),
                Extract::new("pressure"),
            ],
            vec![Extract::new("weather_prediction")],
        );
        let path = create_temp_path();

        config.save(&path).expect("Failed to save");
        let loaded = TrainConfig::load(&path).expect("Failed to load");

        assert_eq!(config, loaded);

        // Cleanup
        let _ = fs::remove_file(&path);
    }
}
