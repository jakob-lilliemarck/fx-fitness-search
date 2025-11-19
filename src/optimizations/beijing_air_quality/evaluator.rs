use super::ingestion::PATHS;
use super::phenotype::BeijingPhenotype;
use crate::core::preprocessor::Transform;
use crate::core::preprocessor::{Cos, Node, Pipeline, Sin};
use crate::core::train_config::TrainConfig;
use futures::future::BoxFuture;
use fx_durable_ga::models::{Evaluator, Terminated};
use serde::{Deserialize, Serialize};
use sqlx::types::Uuid;
use std::io::Write;
use std::process::{Command, Stdio};

pub struct BeijingEvaluator;

impl BeijingEvaluator {
    pub fn new(_model_save_path: &str) -> Self {
        Self
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestVariables {
    prediction_horizon: usize,
    targets: Vec<String>,
}

impl RequestVariables {
    pub fn new(prediction_horizon: usize, targets: Vec<Transform>) -> Self {
        let targets = targets
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>();

        Self {
            prediction_horizon,
            targets,
        }
    }
}

/// Request sent to train_worker subprocess
#[derive(Debug, Serialize)]
struct TrainWorkerRequest {
    train_config: TrainConfig,
    genotype_id: String,
    data_paths: Vec<String>,
    features: Vec<String>,
    targets: Vec<String>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

/// Response from train_worker subprocess
#[derive(Debug, Deserialize)]
struct TrainWorkerResponse {
    fitness: f64,
    genotype_id: String,
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

            // Build time features (always included) as Transform instances
            let mut features = vec![
                Transform {
                    destination: "hour_sin".to_string(),
                    source: "hour".to_string(),
                    pipeline: Pipeline::new(vec![Node::Sin(Sin::new(24.0))]),
                },
                Transform {
                    destination: "hour_cos".to_string(),
                    source: "hour".to_string(),
                    pipeline: Pipeline::new(vec![Node::Cos(Cos::new(24.0))]),
                },
                Transform {
                    destination: "month_sin".to_string(),
                    source: "month".to_string(),
                    pipeline: Pipeline::new(vec![Node::Sin(Sin::new(12.0))]),
                },
                Transform {
                    destination: "month_cos".to_string(),
                    source: "month".to_string(),
                    pipeline: Pipeline::new(vec![Node::Cos(Cos::new(12.0))]),
                },
            ];

            // Convert phenotype features to Transform instances
            for (i, feat) in phenotype.features().iter().enumerate() {
                let destination = format!("feat_{}", i);
                let source = feat.source.clone();
                let pipeline = feat.to_pipeline();
                features.push(Transform {
                    destination,
                    source,
                    pipeline,
                });
            }

            // Target definition (always TEMP with no preprocessing)
            let targets = vec![Transform {
                destination: "target_temp".to_string(),
                source: "TEMP".to_string(),
                pipeline: Pipeline::new(vec![]),
            }];

            let input_size = 4 + phenotype.features().len();
            let train_config = TrainConfig::new(
                input_size,
                phenotype.hidden_size,
                targets.len(),
                phenotype.sequence_length,
                req_var.prediction_horizon,
                features
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>(),
                req_var.targets.iter().map(|t| t.to_string()).collect(),
            );

            // Prepare request for train_worker subprocess
            let worker_request = TrainWorkerRequest {
                train_config,
                genotype_id: genotype_id.to_string(),
                data_paths: PATHS.iter().map(|s| s.to_string()).collect(),
                features: features.iter().map(|t| t.to_string()).collect(),
                targets: targets.iter().map(|t| t.to_string()).collect(),
                epochs: 25,
                batch_size: 100,
                learning_rate: phenotype.learning_rate,
            };

            tracing::info!(
                message = "Spawning train_worker subprocess",
                genotype_id = genotype_id.to_string()
            );

            // Spawn subprocess on blocking task to avoid blocking the async runtime
            // Use select! to allow cancellation - if the future is cancelled, we'll abort the child process
            let task = tokio::task::spawn_blocking(move || {
                // Spawn train_worker subprocess
                let mut child = Command::new("./target/release/train_worker")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| EvaluationError::WorkerFailed(format!("Failed to spawn worker: {}", e)))?;

                // Send request to worker via stdin
                let json_request = serde_json::to_string(&worker_request)?;
                if let Some(mut stdin) = child.stdin.take() {
                    stdin
                        .write_all(json_request.as_bytes())
                        .map_err(|e| EvaluationError::WorkerFailed(format!("Failed to write to worker: {}", e)))?
                } else {
                    return Err(
                        EvaluationError::WorkerFailed("Failed to connect to worker stdin".to_string()).into(),
                    );
                }

                // Wait for worker to finish and read response
                let output = child
                    .wait_with_output()
                    .map_err(|e| EvaluationError::WorkerFailed(format!("Failed to wait for worker: {}", e)))?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(EvaluationError::WorkerFailed(format!(
                        "Worker failed with status {}: {}",
                        output.status, stderr
                    ))
                    .into());
                }

                // Parse response
                let response: TrainWorkerResponse = serde_json::from_slice(&output.stdout)
                    .map_err(|e| EvaluationError::WorkerFailed(format!("Failed to parse worker response: {}", e)))?;

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
                message = "Train worker completed",
                genotype_id = response.genotype_id,
                fitness = response.fitness
            );

            Ok(response.fitness)
        })
    }
}
