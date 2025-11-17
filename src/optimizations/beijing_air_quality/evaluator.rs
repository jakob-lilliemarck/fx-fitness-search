use futures::future::BoxFuture;
use fx_durable_ga::models::{Evaluator, Terminated};
use sqlx::types::Uuid;

use super::dataset::{SequenceDataset, build_dataset_from_file};
use super::ingestion::PATHS;
use super::model::{FeedForward, SequenceModel};
use super::phenotype::BeijingPhenotype;
use super::preprocessor::{Cos, Node, Pipeline, Sin};
use super::train;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::Dataset;

pub struct BeijingEvaluator;

impl Evaluator<BeijingPhenotype> for BeijingEvaluator {
    fn fitness<'a>(
        &self,
        _genotype_id: Uuid,
        phenotype: BeijingPhenotype,
        _terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
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
                            if let (Some(item), Some(metadata)) =
                                (valid.get(i), valid.get_metadata(i))
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
                let input_size = 11; // 4 time features + 7 optimized features
                let model = FeedForward::<Backend>::new(
                    &device,
                    input_size,
                    phenotype.hidden_size,
                    1, // output_size = 1 (single target)
                    phenotype.sequence_length,
                );

            // Train the model
            let (_trained_model, best_valid_loss) = train::train(
                &device,
                &train_dataset,
                &valid_dataset,
                25,  // epochs (fixed)
                100, // batch_size (fixed)
                phenotype.learning_rate,
                model,
                None, // model_save_path (don't save during GA optimization)
                None, // train_config (don't save config during GA)
            )
            .await;

            // Return validation loss as fitness (cast f32 to f64)
            Ok(best_valid_loss as f64)
        })
    }
}
