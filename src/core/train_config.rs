use crate::core::ingestion::Extract;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Validation error: {0}")]
    Validation(String),
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
    /// Epoch at which to start validation. None = validate every epoch starting from epoch 0
    pub validation_start_epoch: Option<usize>,
    /// Number of epochs to train for
    pub epochs: usize,
    /// Batch size for training and validation
    pub batch_size: usize,
    /// Learning rate for the optimizer
    pub learning_rate: f64,
}

impl TrainConfig {
    pub fn new(
        hidden_size: usize,
        sequence_length: usize,
        prediction_horizon: usize,
        features: Vec<Extract>,
        targets: Vec<Extract>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) -> Result<Self, Error> {
        Self::validate_sequence_length(&sequence_length)?;
        Self::validate_hidden_size(&hidden_size)?;
        Self::validate_features(&features)?;
        Self::validate_targets(&targets)?;
        Self::validate_epochs(&epochs)?;
        Self::validate_batch_size(&batch_size)?;
        Self::validate_learning_rate(&learning_rate)?;

        Ok(Self {
            hidden_size,
            sequence_length,
            prediction_horizon,
            features,
            targets,
            weight_decay: None,
            grad_clip: None,
            patience: None,
            validation_start_epoch: None,
            epochs,
            batch_size,
            learning_rate,
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

    /// Builder method to set validation start epoch
    pub fn with_validation_start_epoch(mut self, epoch: usize) -> Result<Self, Error> {
        Self::validate_validation_start_epoch(&epoch, &self.epochs)?;
        self.validation_start_epoch = Some(epoch);
        Ok(self)
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

    fn validate_epochs(epochs: &usize) -> Result<(), Error> {
        if *epochs < 1 {
            return Err(Error::Validation(
                "epochs must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_batch_size(batch_size: &usize) -> Result<(), Error> {
        if *batch_size < 1 {
            return Err(Error::Validation(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_learning_rate(learning_rate: &f64) -> Result<(), Error> {
        if *learning_rate <= 0.0 {
            return Err(Error::Validation(
                "learning_rate must be greater than 0.0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_validation_start_epoch(start_epoch: &usize, epochs: &usize) -> Result<(), Error> {
        if start_epoch >= epochs {
            return Err(Error::Validation(format!(
                "validation_start_epoch ({}) must be less than epochs ({})",
                start_epoch, epochs
            )));
        }
        Ok(())
    }

    /// Parse from a JSON string and validate before returning
    pub fn from_json_str(json: &str) -> Result<Self, Error> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate all fields of this config (use after deserialization)
    fn validate(&self) -> Result<(), Error> {
        // Required fields
        Self::validate_sequence_length(&self.sequence_length)?;
        Self::validate_hidden_size(&self.hidden_size)?;
        Self::validate_features(&self.features)?;
        Self::validate_targets(&self.targets)?;
        Self::validate_epochs(&self.epochs)?;
        Self::validate_batch_size(&self.batch_size)?;
        Self::validate_learning_rate(&self.learning_rate)?;

        // Optional fields
        if let Some(wd) = &self.weight_decay {
            Self::validate_weight_decay(wd)?;
        }
        if let Some(gc) = &self.grad_clip {
            Self::validate_grad_clipping(gc)?;
        }
        if let Some(p) = &self.patience {
            Self::validate_patience(p)?;
        }
        if let Some(se) = &self.validation_start_epoch {
            Self::validate_validation_start_epoch(se, &self.epochs)?;
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
            10,    // epochs
            32,    // batch_size
            0.001, // learning_rate
        )
        .expect("Failed to create test config")
    }

    #[test]
    fn test_new_creates_config_with_correct_values() {
        let config = TrainConfig::new(
            32,
            20,
            2,
            vec![Extract::new("f1")],
            vec![Extract::new("t1")],
            10,
            32,
            0.001,
        )
        .expect("Failed to create config");

        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.sequence_length, 20);
        assert_eq!(config.prediction_horizon, 2);
        assert_eq!(config.features.len(), 1);
        assert_eq!(config.targets.len(), 1);
    }

    #[test]
    fn test_zero_hidden_size_returns_error() {
        let result = TrainConfig::new(
            0,
            10,
            1,
            vec![Extract::new("f1")],
            vec![Extract::new("t1")],
            10,
            32,
            0.001,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_sequence_length_returns_error() {
        let result = TrainConfig::new(
            32,
            0,
            1,
            vec![Extract::new("f1")],
            vec![Extract::new("t1")],
            10,
            32,
            0.001,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_features_returns_error() {
        let result = TrainConfig::new(32, 10, 1, vec![], vec![Extract::new("t1")], 10, 32, 0.001);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_targets_returns_error() {
        let result = TrainConfig::new(32, 10, 1, vec![Extract::new("f1")], vec![], 10, 32, 0.001);
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

    #[test]
    fn test_from_json_str_valid_config() {
        let config = create_test_config();
        let json_str = serde_json::to_string(&config).expect("Failed to serialize");
        let parsed = TrainConfig::from_json_str(&json_str);
        assert!(parsed.is_ok());
        let parsed_cfg = parsed.unwrap();
        assert_eq!(parsed_cfg.hidden_size, config.hidden_size);
        assert_eq!(parsed_cfg.sequence_length, config.sequence_length);
    }

    #[test]
    fn test_from_json_str_invalid_json() {
        let result = TrainConfig::from_json_str("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_str_with_optional_fields() {
        let config = create_test_config()
            .with_weight_decay(1e-3)
            .and_then(|c| c.with_grad_clip(0.5))
            .and_then(|c| c.with_patience(5))
            .expect("Failed to build config");

        let json_str = serde_json::to_string(&config).expect("Failed to serialize");
        let parsed = TrainConfig::from_json_str(&json_str);
        assert!(parsed.is_ok());
        let parsed_cfg = parsed.unwrap();
        assert_eq!(parsed_cfg.weight_decay, Some(1e-3));
        assert_eq!(parsed_cfg.grad_clip, Some(0.5));
        assert_eq!(parsed_cfg.patience, Some(5));
    }
}
