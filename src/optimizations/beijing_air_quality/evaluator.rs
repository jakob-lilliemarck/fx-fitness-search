use super::phenotype::BeijingPhenotype;
use super::protocol::{Request, RequestVariables, Response};
use crate::core::ingestion::Extract;
use crate::core::interpolation::{Interpolation, LinearInterpolator};
use crate::core::preprocessing::{Cos, Node, Sin};
use crate::core::train_config::TrainConfig;
use futures::future::BoxFuture;
use fx_durable_ga::models::{Evaluator, Terminated};
use sqlx::types::Uuid;
use std::io::Write;
use std::process::{Command, Stdio};
use uuid::Uuid as GuidUuid;

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("Missing request data")]
    MissingData,
    #[error("Worker subprocess failed: {0}")]
    WorkerFailed(String),
}

/// The evaluator, registered to the GA evaluator registry
/// This is a shared runtime configurable struct.
pub struct BeijingEvaluator {
    model_save_path: Option<String>,
}

impl BeijingEvaluator {
    pub fn new(model_save_path: &str) -> Self {
        Self {
            model_save_path: Some(model_save_path.to_string()),
        }
    }
}

fn append_time_transform(extracts: &mut Vec<Extract>, key: &str, period: u32) {
    extracts.push(Extract::new(key).with_node(Node::Sin(Sin::new(period as f32))));
    extracts.push(Extract::new(key).with_node(Node::Cos(Cos::new(period as f32))));
}

impl Evaluator<BeijingPhenotype> for BeijingEvaluator {
    fn fitness<'a>(
        &self,
        genotype_id: Uuid,
        phenotype: BeijingPhenotype,
        request: &'a fx_durable_ga::models::Request,
        _terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        let model_save_path = self.model_save_path.clone();
        Box::pin(async move {
            // Deserialize RequestVariables from the request.data field.
            let req_var = serde_json::from_value::<RequestVariables>(
                request.data.clone().ok_or(EvaluationError::MissingData)?,
            )?;

            // Feature transforms
            let mut features = phenotype.features().to_vec();
            let mut targets = Vec::new();

            // Append time transforms to features transforms
            append_time_transform(&mut features, "month", 12);
            append_time_transform(&mut features, "day_of_week", 7);
            append_time_transform(&mut features, "hour", 24);

            // Append a single target to predict
            targets.push(
                Extract::new("TEMP")
                    .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            );

            // Construct the TrainConfig  as expected by sync_train()
            let train_config = TrainConfig::new(
                phenotype.hidden_size,
                phenotype.sequence_length,
                req_var.prediction_horizon,
                features,
                targets,
            );

            // Prepare request for training binary
            let train_request = Request {
                genotype_id: GuidUuid::from_bytes(*genotype_id.as_bytes()),
                train_config,
                epochs: 25,
                batch_size: 100,
                learning_rate: phenotype.learning_rate,
                model_save_path: model_save_path
                    .map(|p| format!("{}/{}", p, genotype_id.to_string())),
            };

            // Store the request json document
            if let Some(path) = train_request
                .model_save_path
                .as_ref()
                .map(|p| format!("{}.request.json", p))
            {
                train_request.save(&path)?;
            }

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
                let response = serde_json::from_slice::<Response>(&output.stdout).map_err(|e| {
                    EvaluationError::WorkerFailed(format!("Failed to parse binary response: {}", e))
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
