use super::phenotype::BeijingPhenotype;
use crate::core::ingestion::Extract;
use crate::core::train_config::TrainConfig;
use futures::future::BoxFuture;
use fx_durable_ga::models::{Evaluator, Terminated};
use serde::{Deserialize, Serialize};
use sqlx::types::Uuid;
use std::io::Write;
use std::process::{Command, Stdio};
use uuid::Uuid as GuidUuid;

pub struct BeijingEvaluator;

impl BeijingEvaluator {
    pub fn new(_model_save_path: &str) -> Self {
        Self
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestVariables {
    pub prediction_horizon: usize,
}

impl RequestVariables {
    pub fn new(prediction_horizon: usize) -> Self {
        Self { prediction_horizon }
    }
}

/// Request sent to training binary
#[derive(Debug, Serialize)]
struct TrainRequest {
    genotype_id: GuidUuid,
    train_config: TrainConfig,
    processors_features: Vec<Extract>,
    processors_targets: Vec<Extract>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

/// Response from training binary
#[derive(Debug, Deserialize)]
struct TrainResponse {
    fitness: f64,
    genotype_id: GuidUuid,
}

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("Missing request data")]
    MissingData,
    #[error("Worker subprocess failed: {0}")]
    WorkerFailed(String),
}

impl Evaluator<BeijingPhenotype> for BeijingEvaluator {
    fn fitness<'a>(
        &self,
        genotype_id: Uuid,
        phenotype: BeijingPhenotype,
        request: &'a fx_durable_ga::models::Request,
        _terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            let req_var: RequestVariables =
                serde_json::from_value(request.data.clone().ok_or(EvaluationError::MissingData)?)?;

            tracing::info!(message = "Request data", req_var = ?req_var);

            // Get feature Extracts directly from phenotype
            let processors_features = phenotype.features().to_vec();

            // Target and time transforms are hardcoded in the binary
            // Here we just send an empty vec for targets
            let processors_targets = Vec::new();

            let train_config = TrainConfig::new(
                processors_features.len(),
                phenotype.hidden_size,
                1, // output_size - always 1 target
                phenotype.sequence_length,
                req_var.prediction_horizon,
                processors_features
                    .iter()
                    .map(|e| e.source.clone())
                    .collect::<Vec<String>>(),
                vec!["TEMP".to_string()],
            );

            // Prepare request for training binary
            let train_request = TrainRequest {
                genotype_id: GuidUuid::from_bytes(*genotype_id.as_bytes()),
                train_config,
                processors_features,
                processors_targets,
                epochs: 25,
                batch_size: 100,
                learning_rate: phenotype.learning_rate,
            };

            tracing::info!(
                message = "Spawning training binary",
                genotype_id = genotype_id.to_string()
            );

            // Spawn subprocess on blocking task to avoid blocking the async runtime
            let task = tokio::task::spawn_blocking(move || {
                // Spawn training binary
                let mut child = Command::new("./target/release/beijing")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| {
                        EvaluationError::WorkerFailed(format!("Failed to spawn binary: {}", e))
                    })?;

                // Send request to binary via stdin
                let json_request = serde_json::to_string(&train_request)?;
                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(json_request.as_bytes()).map_err(|e| {
                        EvaluationError::WorkerFailed(format!("Failed to write to binary: {}", e))
                    })?
                } else {
                    return Err(EvaluationError::WorkerFailed(
                        "Failed to connect to binary stdin".to_string(),
                    )
                    .into());
                }

                // Wait for binary to finish and read response
                let output = child.wait_with_output().map_err(|e| {
                    EvaluationError::WorkerFailed(format!("Failed to wait for binary: {}", e))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(EvaluationError::WorkerFailed(format!(
                        "Binary failed with status {}: {}",
                        output.status, stderr
                    ))
                    .into());
                }

                // Parse response
                let response: TrainResponse =
                    serde_json::from_slice(&output.stdout).map_err(|e| {
                        EvaluationError::WorkerFailed(format!(
                            "Failed to parse binary response: {}",
                            e
                        ))
                    })?;

                Ok(response)
            });

            let response = task
                .await
                .map_err(|e| {
                    if e.is_cancelled() {
                        // Task was cancelled - this is expected if the fitness evaluation was aborted
                        tracing::info!("Training task was cancelled");
                    }
                    EvaluationError::WorkerFailed(format!("Task failed: {}", e))
                })?
                .map_err(|e: anyhow::Error| e)?;

            tracing::info!(
                message = "Training completed",
                genotype_id = response.genotype_id.to_string(),
                fitness = response.fitness
            );

            Ok(response.fitness)
        })
    }
}
