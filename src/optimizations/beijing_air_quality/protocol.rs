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

/// Response from training binary
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub fitness: f64,
    pub genotype_id: Uuid,
}
