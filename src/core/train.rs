use super::batcher::SequenceBatcher;
use super::model::SequenceModel;
use super::train_config::TrainConfig;
use crate::core::dataset::SequenceDatasetItem;
use burn::data::dataloader::Dataset;
use burn::data::dataloader::batcher::Batcher;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;

pub fn train_sync<B, M>(
    device: &B::Device,
    dataset_training: &impl Dataset<SequenceDatasetItem>,
    dataset_validation: &impl Dataset<SequenceDatasetItem>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    mut model: M,
    model_save_path: Option<String>,
    train_config: Option<TrainConfig>,
) -> (M, f32)
where
    B: AutodiffBackend,
    M: SequenceModel<B> + AutodiffModule<B>,
    M::InnerModule: SequenceModel<B::InnerBackend>,
{
    // Initialize optimizer
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Create batchers
    let batcher_train = SequenceBatcher::<B>::new();
    let batcher_valid = SequenceBatcher::<B::InnerBackend>::new();

    let dataset_train_len = dataset_training.len();
    let dataset_valid_len = dataset_validation.len();

    // Early stopping variables
    let mut best_valid_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0;
    let patience = 10;

    for epoch in 0..epochs {
        // ============ Training Loop ============
        let mut total_train_loss = 0.0;
        let mut num_train_batches = 0;

        // step_by(batch_size) generates starting indices: 0, batch_size, 2*batch_size, ...
        // This partitions the dataset into non-overlapping groups of items.
        // Each group becomes one batch for the forward/backward pass.
        for start_idx in (0..dataset_train_len).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(dataset_train_len);

            // Collect batch_size items (or fewer for the last partial batch)
            // Each item is a single training example: (sequence, target) pair
            let items: Vec<_> = (start_idx..end_idx)
                .filter_map(|i| dataset_training.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            // Batcher converts the Vec of items into tensor batch:
            // - sequences: Tensor<B, 3> of shape [batch_size, seq_len, features]
            // - targets: Tensor<B, 2> of shape [batch_size, output_size]
            let batch = batcher_train.batch(items, device);

            // Forward pass
            let outputs = model.forward(batch.sequences);

            // MSE Loss
            let loss = (outputs - batch.targets).powf_scalar(2.0).mean();

            // Extract scalar BEFORE backward to avoid keeping the loss tensor
            let loss_value = loss.clone().into_scalar().elem::<f32>();

            // Backward pass - this creates the gradient graph
            let grads = loss.backward();

            // Convert gradients to params
            let grads_params = GradientsParams::from_grads(grads, &model);

            // Update parameters - this consumes the gradients
            model = optimizer.step(learning_rate, model, grads_params);

            total_train_loss += loss_value;
            num_train_batches += 1;
        }

        let avg_train_loss = if num_train_batches > 0 {
            total_train_loss / num_train_batches as f32
        } else {
            0.0
        };

        // ============ Validation Loop ============
        let valid_model = model.valid();
        let mut total_valid_loss = 0.0;
        let mut num_valid_batches = 0;

        for start_idx in (0..dataset_valid_len).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(dataset_valid_len);

            let items: Vec<_> = (start_idx..end_idx)
                .filter_map(|i| dataset_validation.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher_valid.batch(items, device);

            // Validation forward pass (no gradients needed)
            let loss_value = {
                let outputs = valid_model.forward(batch.sequences);
                let loss = (outputs - batch.targets).powf_scalar(2.0).mean();
                loss.into_scalar().elem::<f32>()
            }; // Tensor dropped here

            total_valid_loss += loss_value;
            num_valid_batches += 1;
        }

        let avg_valid_loss = if num_valid_batches > 0 {
            total_valid_loss / num_valid_batches as f32
        } else {
            0.0
        };

        // ============ Early Stopping Check ============
        if avg_valid_loss < best_valid_loss {
            best_valid_loss = avg_valid_loss;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= patience {
                tracing::info!(
                    epoch = epoch + 1,
                    best_valid_loss = best_valid_loss,
                    "Early stopping triggered",
                );
                break;
            }
        }

        tracing::info!(
            epoch = epoch + 1,
            total_epochs = epochs,
            train_loss = avg_train_loss,
            valid_loss = avg_valid_loss,
            best_valid_loss = best_valid_loss,
            "Epoch completed",
        );
    }

    // Save the trained model and config if a path was provided
    if let Some(path) = model_save_path {
        model
            .clone()
            .save_file(&path, &CompactRecorder::new())
            .expect("Failed to save model");

        // Save config alongside the model if provided
        if let Some(config) = train_config {
            let config_path = format!("{}.config.json", path);
            config.save(&config_path).expect("Failed to save config");
        }
    }

    (model, best_valid_loss)
}
