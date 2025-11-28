use crate::core::ingestion::Extract;
use serde::{Deserialize, Serialize};
use std::{fs, io};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

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
    ) -> Result<Self, Error> {
        Self::validate_sequence_length(&sequence_length)?;
        Self::validate_hidden_size(&hidden_size)?;
        Self::validate_features(&features)?;
        Self::validate_targets(&targets)?;

        Ok(Self {
            hidden_size,
            sequence_length,
            prediction_horizon,
            features,
            targets,
            weight_decay: None,
            grad_clip: None,
            patience: None,
        })
    }

    /// Builder method to set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self, Error> {
        Self::validate_weight_decay(&weight_decay)?;
        self.weight_decay = Some(weight_decay);
        Ok(self)
    }

    /// Builder method to set gradient clipping
    pub fn with_grad_clip(mut self, grad_clip: f32) -> Result<Self, Error> {
        Self::validate_grad_clipping(&grad_clip)?;
        self.grad_clip = Some(grad_clip);
        Ok(self)
    }

    /// Builder method to set early stopping patience
    pub fn with_patience(mut self, patience: usize) -> Result<Self, Error> {
        Self::validate_patience(&patience)?;
        self.patience = Some(patience);
        Ok(self)
    }

    pub fn save(&self, path: &str) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Error> {
        let json = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&json)?;

        // Validate required fields
        Self::validate_sequence_length(&config.sequence_length)?;
        Self::validate_hidden_size(&config.hidden_size)?;
        Self::validate_features(&config.features)?;
        Self::validate_targets(&config.targets)?;

        // Validate optional fields using map
        config
            .weight_decay
            .as_ref()
            .map(|wd| Self::validate_weight_decay(wd))
            .transpose()?;
        config
            .grad_clip
            .as_ref()
            .map(|gc| Self::validate_grad_clipping(gc))
            .transpose()?;
        config
            .patience
            .as_ref()
            .map(|p| Self::validate_patience(p))
            .transpose()?;

        Ok(config)
    }

    fn validate_sequence_length(sequence_length: &usize) -> Result<(), Error> {
        if *sequence_length < 1 {
            return Err(Error::Validation(
                "sequence_length must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_hidden_size(hidden_size: &usize) -> Result<(), Error> {
        if *hidden_size < 1 {
            return Err(Error::Validation(
                "hidden_size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_patience(patience: &usize) -> Result<(), Error> {
        if *patience < 1 {
            return Err(Error::Validation(
                "patience must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_grad_clipping(grad_clipping: &f32) -> Result<(), Error> {
        if *grad_clipping <= 0.0 {
            return Err(Error::Validation(
                "grad_clipping must be greater than 0.0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_weight_decay(weight_decay: &f32) -> Result<(), Error> {
        if *weight_decay <= 0.0 {
            return Err(Error::Validation(
                "weight_decay must be greater than 0.0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_targets(targets: &[Extract]) -> Result<(), Error> {
        if targets.is_empty() {
            return Err(Error::Validation("targets must not be empty".to_string()));
        }
        Ok(())
    }

    fn validate_features(features: &[Extract]) -> Result<(), Error> {
        if features.is_empty() {
            return Err(Error::Validation("features must not be empty".to_string()));
        }
        Ok(())
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
        .expect("Failed to create test config")
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
        )
        .expect("Failed to create config");

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
        )
        .expect("Failed to create config");
        let path = create_temp_path();

        config.save(&path).expect("Failed to save");
        let loaded = TrainConfig::load(&path).expect("Failed to load");

        assert_eq!(config, loaded);

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_zero_hidden_size_returns_error() {
        let result = TrainConfig::new(0, 10, 1, vec![Extract::new("f1")], vec![Extract::new("t1")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_sequence_length_returns_error() {
        let result = TrainConfig::new(32, 0, 1, vec![Extract::new("f1")], vec![Extract::new("t1")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_features_returns_error() {
        let result = TrainConfig::new(32, 10, 1, vec![], vec![Extract::new("t1")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_targets_returns_error() {
        let result = TrainConfig::new(32, 10, 1, vec![Extract::new("f1")], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_weight_decay_returns_error() {
        let config = create_test_config();
        let result = config.with_weight_decay(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_grad_clip_returns_error() {
        let config = create_test_config();
        let result = config.with_grad_clip(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_patience_returns_error() {
        let config = create_test_config();
        let result = config.with_patience(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_builder_methods() {
        let config = create_test_config()
            .with_weight_decay(1e-3)
            .and_then(|c| c.with_grad_clip(0.5))
            .and_then(|c| c.with_patience(5))
            .expect("Failed to build config");

        assert_eq!(config.weight_decay, Some(1e-3));
        assert_eq!(config.grad_clip, Some(0.5));
        assert_eq!(config.patience, Some(5));
    }
}
