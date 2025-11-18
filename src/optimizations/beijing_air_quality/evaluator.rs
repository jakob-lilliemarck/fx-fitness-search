use super::ingestion::PATHS;
use super::ingestion::build_dataset_from_file;
use super::phenotype::BeijingPhenotype;
use crate::core::dataset::SequenceDataset;
use crate::core::model::{FeedForward, SequenceModel};
use crate::core::preprocessor::{Cos, Node, Pipeline, Sin};
use crate::core::train;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::Dataset;
use futures::future::BoxFuture;
use fx_durable_ga::models::{Evaluator, Terminated};
use serde::Deserialize;
use serde::Serialize;
use sqlx::types::Uuid;

pub struct BeijingEvaluator {
    model_save_path: String,
}

impl BeijingEvaluator {
    pub fn new(model_save_path: &str) -> Self {
        Self {
            model_save_path: model_save_path.to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestVariables {
    prediction_horizon: usize,
    targets: String,
}

impl RequestVariables {
    pub fn new(prediction_horizon: usize, targets: String) -> Self {
        Self {
            prediction_horizon,
            targets,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("Missing request data")]
    MissingData,
}

impl Evaluator<BeijingPhenotype> for BeijingEvaluator {
    fn fitness<'a>(
        &self,
        genotype_id: Uuid,
        phenotype: BeijingPhenotype,
        request: &'a fx_durable_ga::models::Request,
        _terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        // The request is the place where we should store prediction_horizon and targets (incl pipelines)
        // It will be a json field on request - just deserialize it and use in this function.

        let model_save_path = self.model_save_path.clone();

        Box::pin(async move {
            let req_var: RequestVariables =
                serde_json::from_value(request.data.clone().ok_or(EvaluationError::MissingData)?)?;
            tracing::info!(message = "Request data", req_var = ?req_var);

            type Backend = Autodiff<NdArray>;
            let device = NdArrayDevice::default();

            // Build time features (always included)
            let mut features = vec![
                (
                    "hour_sin".to_string(),
                    "hour".to_string(),
                    Pipeline::new(vec![Node::Sin(Sin::new(24.0))]),
                ),
                (
                    "hour_cos".to_string(),
                    "hour".to_string(),
                    Pipeline::new(vec![Node::Cos(Cos::new(24.0))]),
                ),
                (
                    "month_sin".to_string(),
                    "month".to_string(),
                    Pipeline::new(vec![Node::Sin(Sin::new(12.0))]),
                ),
                (
                    "month_cos".to_string(),
                    "month".to_string(),
                    Pipeline::new(vec![Node::Cos(Cos::new(12.0))]),
                ),
            ];

            // Convert phenotype features to feature definitions
            for (i, feat) in phenotype.features().iter().enumerate() {
                let name = format!("feat_{}", i);
                let source = feat.source.clone();
                let pipeline = feat.to_pipeline();
                features.push((name, source, pipeline));
            }

            // Target definition (always TEMP with no preprocessing)
            let targets = vec![(
                "target_temp".to_string(),
                "TEMP".to_string(),
                Pipeline::new(vec![]),
            )];

            // Load data from all stations and combine into unified datasets
            let mut all_train_items = Vec::new();
            let mut all_valid_items = Vec::new();

            for path in PATHS {
                let builder = build_dataset_from_file(path, &features, &targets)?;

                // Split 80/20 train/validation
                let (train_dataset, valid_dataset) = builder.build(
                    phenotype.sequence_length,
                    1, // prediction_horizon = 1 (predict 1 timestep ahead)
                    Some(0.8),
                )?;

                // Extract items with metadata from training dataset
                for i in 0..train_dataset.len() {
                    if let (Some(item), Some(metadata)) =
                        (train_dataset.get(i), train_dataset.get_metadata(i))
                    {
                        all_train_items.push((metadata.clone(), item));
                    }
                }

                // Extract items with metadata from validation dataset
                if let Some(valid) = valid_dataset {
                    for i in 0..valid.len() {
                        if let (Some(item), Some(metadata)) = (valid.get(i), valid.get_metadata(i))
                        {
                            all_valid_items.push((metadata.clone(), item));
                        }
                    }
                }
            }

            // Combine all items into unified datasets
            let train_dataset = SequenceDataset::from_items(all_train_items);
            let valid_dataset = SequenceDataset::from_items(all_valid_items);

            // Create FeedForward model
            let input_size = 4 + phenotype.features().len();
            let model = FeedForward::<Backend>::new(
                &device,
                input_size,
                phenotype.hidden_size,
                targets.len(),
                phenotype.sequence_length,
            );

            // Create a unique filepath to store the model and config files of this genotype
            let model_file_path = format!("{}/{}", model_save_path, genotype_id.to_string());

            // Train the model
            let (_trained_model, best_valid_loss) = train::train(
                &device,
                &train_dataset,
                &valid_dataset,
                25,  // epochs (fixed)
                100, // batch_size (fixed)
                phenotype.learning_rate,
                model,
                Some(model_file_path), // model_save_path (don't save during GA optimization)
                None,                  // train_config (don't save config during GA)
            )
            .await;

            // Return validation loss as fitness (cast f32 to f64)
            Ok(best_valid_loss as f64)
        })
    }
}
