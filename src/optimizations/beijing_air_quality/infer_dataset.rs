//! Inference evaluation over historical datasets.
//!
//! ## Performance Note: Batched Inference Opportunity
//!
//! Currently, this implementation processes rows sequentially:
//! 1. Read CSV row by row
//! 2. Apply pipelines to each row
//! 3. Make individual predictions (35k+ forward passes)
//!
//! For large datasets (35k rows), this can be slow (~1000+ forward passes).
//! Since we're operating on historical data, we could optimize by:
//!
//! **Proposed refactor:**
//! 1. First pass: Read all CSV rows, apply all pipelines → collect all_features/all_targets
//! 2. Second pass: Create all sequences at once (use SequenceDataset from dataset.rs)
//! 3. Third pass: Batch inference using SequenceBatcher (32 sequences/batch → ~1000 total batches)
//! 4. Match predictions back to row numbers
//!
//! Expected speedup: ~35x (from 35k single forward passes → 1k batched forward passes).
//! Tradeoff: Requires loading all preprocessed data into memory.

#![allow(dead_code)]

use super::ingestion::build_dataset_from_file;
use crate::core::dataset::Metadata;
use crate::core::inference::InferenceEngine;
use crate::core::preprocessor::Transform;
use burn::data::dataloader::Dataset;
use burn::prelude::Backend;
use std::fmt;
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub metadata: Metadata, // Source sequence metadata (start, end, target row numbers)
    pub target: Vec<f32>,   // The actual value at the prediction target timestep
    pub prediction: Vec<f32>, // Model prediction
    pub prediction_naive: Vec<f32>, // Naive prediction using the value of the current timestep
    pub dist_prediction: Vec<f32>, // |target - prediction|
    pub dist_prediction_naive: Vec<f32>, // |target - prediction_naive|
    pub baseline_accuracy: Vec<f32>, // dist_prediction_naive / dist_prediction if prediction is better, else 0
}

impl fmt::Display for InferenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] pred: {:?}, actual: {:?}, current: {:?}",
            self.metadata.target_row_id,
            format_vec(&self.prediction),
            format_vec(&self.target),
            format_vec(&self.prediction_naive)
        )
    }
}

fn format_vec(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>()
        .join(", ")
}

pub struct InferenceMetrics {
    pub mse: f32,
    pub rmse: f32,
    pub count: usize,
    pub mean_accuracy: f32,
    pub median_accuracy: f32,
    pub pct_above_baseline: f32, // % predictions > 0%
    pub pct_good: f32,           // % predictions > 50%
}

impl fmt::Display for InferenceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Predictions: {}, MSE: {:.6}, RMSE: {:.6}, Mean: {:.2}%, Median: {:.2}%, >Baseline: {:.1}%, >50%: {:.1}%",
            self.count,
            self.mse,
            self.rmse,
            self.mean_accuracy,
            self.median_accuracy,
            self.pct_above_baseline,
            self.pct_good
        )
    }
}

/// Run inference on a dataset using the same pipeline infrastructure as training
///
/// If `limit` is specified, stops after generating that many predictions.
/// If `limit` is None, processes the entire dataset.
pub fn run_inference<B: Backend>(
    engine: &InferenceEngine<B>,
    dataset_path: &str,
    limit: Option<usize>,
) -> anyhow::Result<Vec<InferenceResult>> {
    // Parse feature and target definitions from engine.config
    let features = parse_definitions(&engine.config.features)?;
    let targets = parse_definitions(&engine.config.targets)?;

    let sequence_length = engine.config.sequence_length;
    let prediction_horizon = engine.config.prediction_horizon;

    // Use DatasetBuilder to load and preprocess the dataset
    // This reuses the exact same pipeline infrastructure as training
    let dataset_builder = build_dataset_from_file(dataset_path, &features, &targets)?;

    // Build without splitting (None for split parameter means no train/validation split)
    let (inference_dataset, _) =
        dataset_builder.build(sequence_length, prediction_horizon, None)?;

    let mut results = Vec::new();

    // Iterate through all items in the dataset
    for idx in 0..inference_dataset.len() {
        // Get the metadata and item for this sequence
        let metadata = match inference_dataset.get_metadata(idx) {
            Some(m) => m,
            None => {
                tracing::debug!("No item at index: [{}]", idx);
                continue;
            }
        };

        let item = inference_dataset
            .get(idx)
            .expect("item should exist if metadata exists");

        // The naive prediction is the preprocessed target value at the sequence end (the anchor point)
        let prediction_naive = metadata.target_at_sequence_end.clone();

        // Make prediction
        match engine.predict(item.sequence) {
            Ok(prediction) => {
                // Calculate distance metrics
                let dist_baseline: Vec<f32> = item
                    .target
                    .iter()
                    .zip(prediction_naive.iter())
                    .map(|(a, c)| (a - c).abs())
                    .collect();

                let dist_prediction: Vec<f32> = item
                    .target
                    .iter()
                    .zip(prediction.iter())
                    .map(|(a, p)| (a - p).abs())
                    .collect();

                let accuracy: Vec<f32> = dist_prediction
                    .iter()
                    .zip(dist_baseline.iter())
                    .map(|(dist_pred, dist_base)| {
                        if dist_pred > dist_base {
                            0.0
                        } else if dist_base == &0.0 {
                            if dist_pred == &0.0 { 100.0 } else { 0.0 }
                        } else {
                            (1.0 - (dist_pred / dist_base)) * 100.0
                        }
                    })
                    .collect();

                let result = InferenceResult {
                    metadata: metadata.clone(),
                    prediction,
                    target: item.target,
                    prediction_naive,
                    dist_prediction_naive: dist_baseline,
                    dist_prediction,
                    baseline_accuracy: accuracy,
                };
                results.push(result);

                // Check if we've reached the limit
                if let Some(max_predictions) = limit {
                    if results.len() >= max_predictions {
                        return Ok(results);
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: inference failed at sequence {}: {}", idx, e);
            }
        }
    }

    Ok(results)
}

/// Parse feature/target definitions back into Transform instances
fn parse_definitions(definitions: &[String]) -> anyhow::Result<Vec<Transform>> {
    let mut result = Vec::new();

    for def in definitions {
        // Use Transform::try_from for consistent parsing
        let transform = Transform::try_from(def.as_str())
            .map_err(|e| anyhow::anyhow!("Invalid transform definition '{}': {}", def, e))?;
        result.push(transform);
    }

    Ok(result)
}

/// Write inference results to CSV file
pub fn write_results_to_csv(results: &[InferenceResult], path: &str) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // Write header with metadata columns
    writeln!(
        file,
        "sequence_start_row_no,sequence_end_row_no,target_row_no,prediction,actual_value,current_value,dist_baseline,dist_prediction,accuracy"
    )?;

    // Write data rows
    for result in results {
        let pred_str = result
            .prediction
            .first()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let actual_str = result
            .target
            .first()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let current_str = result
            .prediction_naive
            .first()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let dist_base_str = result
            .dist_prediction_naive
            .first()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let dist_pred_str = result
            .dist_prediction
            .first()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let acc_str = result
            .baseline_accuracy
            .first()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "N/A".to_string());

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            result.metadata.sequence_start_row_id,
            result.metadata.sequence_end_row_id,
            result.metadata.target_row_id,
            pred_str,
            actual_str,
            current_str,
            dist_base_str,
            dist_pred_str,
            acc_str
        )?
    }

    Ok(())
}

/// Calculate MSE, RMSE and accuracy metrics from inference results
pub fn calculate_metrics(results: &[InferenceResult]) -> InferenceMetrics {
    if results.is_empty() {
        return InferenceMetrics {
            mse: 0.0,
            rmse: 0.0,
            count: 0,
            mean_accuracy: 0.0,
            median_accuracy: 0.0,
            pct_above_baseline: 0.0,
            pct_good: 0.0,
        };
    }

    let mut sum_squared_error = 0.0;
    let mut accuracies = Vec::new();

    for result in results {
        // Assume single-value predictions and targets for simplicity
        if !result.prediction.is_empty() && !result.target.is_empty() {
            let pred = result.prediction[0];
            let actual = result.target[0];
            let error = pred - actual;
            sum_squared_error += error * error;

            // Collect accuracy values (assuming single output)
            if !result.baseline_accuracy.is_empty() {
                accuracies.push(result.baseline_accuracy[0]);
            }
        }
    }

    let mse = sum_squared_error / results.len() as f32;
    let rmse = mse.sqrt();

    // Calculate accuracy metrics
    let mean_accuracy = if !accuracies.is_empty() {
        accuracies.iter().sum::<f32>() / accuracies.len() as f32
    } else {
        0.0
    };

    // Sort for median calculation
    accuracies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_accuracy = if !accuracies.is_empty() {
        if accuracies.len() % 2 == 0 {
            (accuracies[accuracies.len() / 2 - 1] + accuracies[accuracies.len() / 2]) / 2.0
        } else {
            accuracies[accuracies.len() / 2]
        }
    } else {
        0.0
    };

    // Count predictions above baseline (> 0%)
    let above_baseline = accuracies.iter().filter(|a| **a > 0.0).count();
    let pct_above_baseline = if !accuracies.is_empty() {
        (above_baseline as f32 / accuracies.len() as f32) * 100.0
    } else {
        0.0
    };

    // Count good predictions (> 50%)
    let good_predictions = accuracies.iter().filter(|a| **a > 50.0).count();
    let pct_good = if !accuracies.is_empty() {
        (good_predictions as f32 / accuracies.len() as f32) * 100.0
    } else {
        0.0
    };

    InferenceMetrics {
        mse,
        rmse,
        count: results.len(),
        mean_accuracy,
        median_accuracy,
        pct_above_baseline,
        pct_good,
    }
}
