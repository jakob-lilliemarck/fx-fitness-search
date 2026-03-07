use super::indexers::BeijingIndexer;
use super::phenotype::{
    BeijingPhenotype, EvaluationError, MutationParams, append_time_transform, decode_user_defined,
};
use super::protocol::{Request, Response};
use crate::core::ingestion::Extract;
use crate::core::interpolation::{Interpolation, LinearInterpolator};
use crate::core::train_config::TrainConfig;
use anyhow::{Context, Result as AnyhowResult};
use futures::future::BoxFuture;
use fx_durable_ga::models::foreign_service;
use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};
use tokio::task;
use uuid::Uuid as GuidUuid;

pub struct BeijingOptimizer {
    model_save_path: Option<String>,
    batch_size: usize,
}

impl BeijingOptimizer {
    pub fn new(model_save_path: &str, batch_size: usize) -> Self {
        Self {
            model_save_path: Some(model_save_path.to_string()),
            batch_size,
        }
    }
}

pub fn new_beijing_service(
    model_save_path: &str,
    batch_size: usize,
) -> foreign_service::OptimizationService<BeijingPhenotype, BeijingOptimizer> {
    foreign_service::OptimizationService::new(
        BeijingPhenotype::TYPE_NAME,
        BeijingOptimizer::new(model_save_path, batch_size),
    )
    .with_indexer(BeijingIndexer)
}

impl foreign_service::Optimizer for BeijingOptimizer {
    type Type = BeijingPhenotype;

    fn random(&self, _user_defined: &Value) -> AnyhowResult<Self::Type> {
        let mut rng = rand::rng();
        Ok(BeijingPhenotype::random(&mut rng))
    }

    fn mutate(
        &self,
        instance: &mut Self::Type,
        user_defined: &serde_json::Value,
    ) -> AnyhowResult<()> {
        let config = decode_user_defined(user_defined)?;

        let params = MutationParams {
            mutation_rate: config.mutation_rate,
            temperature: config.temperature,
        };

        let mut rng = rand::rng();

        instance.mutate(&mut rng, params);
        Ok(())
    }

    fn crossover(
        &self,
        parent1: Self::Type,
        parent2: Self::Type,
        _user_defined: &Value,
    ) -> AnyhowResult<Self::Type> {
        let mut rng = rand::rng();
        Ok(BeijingPhenotype::crossover(&parent1, &parent2, &mut rng))
    }

    fn evaluate<'a>(
        &'a self,
        instance: &'a Self::Type,
        user_defined: &'a Value,
    ) -> BoxFuture<'a, AnyhowResult<f64>> {
        let model_save_path = self.model_save_path.clone();
        let batch_size = self.batch_size;

        Box::pin(async move {
            let config = decode_user_defined(user_defined)?;

            let mut features = instance.features.clone();
            features.push(Extract::new("pos_x"));
            features.push(Extract::new("pos_y"));
            append_time_transform(&mut features, "month", 12);
            append_time_transform(&mut features, "day_of_week", 7);
            append_time_transform(&mut features, "hour", 24);

            let mut targets = Vec::with_capacity(1);
            targets.push(
                Extract::new("TEMP")
                    .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            );

            let train_config = TrainConfig::new(
                instance.hidden_size,
                instance.sequence_length,
                config.training.prediction_horizon,
                features,
                targets,
                config.training.epochs,
                batch_size,
                instance.learning_rate,
            )?
            .with_patience(config.training.patience)?
            .with_validation_start_epoch(config.training.validation_start_epoch)?;

            let genotype_id = GuidUuid::now_v7();
            let model_path = model_save_path
                .as_ref()
                .map(|path| format!("{}/{}", path, genotype_id));

            let train_request = Request {
                genotype_id,
                train_config: serde_json::to_value(&train_config)?,
                model_save_path: model_path,
                batch_size,
            };

            if let Some(path) = train_request
                .model_save_path
                .as_ref()
                .map(|p| format!("{}.request.json", p))
            {
                train_request.save(&path)?;
            }

            tracing::info!(
                message = "Spawning training binary",
                genotype_id = %train_request.genotype_id
            );

            let json_request =
                serde_json::to_string(&train_request).context("Failed to encode train request")?;

            let task = task::spawn_blocking(move || -> AnyhowResult<Response> {
                let mut child = Command::new("./target/release/beijing")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|err| {
                        EvaluationError::WorkerFailed(format!("Failed to spawn binary: {err}"))
                    })?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(json_request.as_bytes()).map_err(|err| {
                        EvaluationError::WorkerFailed(format!("Failed to write to binary: {err}"))
                    })?;
                } else {
                    return Err(EvaluationError::WorkerFailed(
                        "Failed to connect to binary stdin".to_string(),
                    )
                    .into());
                }

                let output = child.wait_with_output().map_err(|err| {
                    EvaluationError::WorkerFailed(format!("Failed to wait for binary: {err}"))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(EvaluationError::WorkerFailed(format!(
                        "Binary failed with status {}: {}",
                        output.status, stderr
                    ))
                    .into());
                }

                serde_json::from_slice::<Response>(&output.stdout).map_err(|err| {
                    EvaluationError::WorkerFailed(format!("Failed to parse binary response: {err}"))
                        .into()
                })
            });

            let response = task.await.map_err(|err| {
                EvaluationError::WorkerFailed(format!("Task failed before completion: {err}"))
            })??;

            tracing::info!(
                message = "Training completed",
                genotype_id = %response.genotype_id,
                fitness = response.fitness
            );

            Ok(response.fitness)
        })
    }
}
