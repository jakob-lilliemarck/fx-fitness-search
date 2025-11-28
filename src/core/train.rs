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

/// Compute average loss by dividing total accumulated loss by number of batches.
/// If no batches were processed, returns 0.0 to avoid division by zero.
fn compute_average_loss(total_loss: f32, num_batches: usize) -> f32 {
    if num_batches > 0 {
        total_loss / num_batches as f32
    } else {
        0.0
    }
}

/// Early stopping state update. Given current and best validation losses,
/// returns (should_stop, new_epochs_without_improvement, new_best_loss).
/// - If current < best: improvement detected; reset counter, update best
/// - If current >= best: no improvement; increment counter
/// - Triggers stop only if patience is Some and counter reaches it
fn update_early_stopping_state(
    best_valid_loss: f32,
    current_valid_loss: f32,
    epochs_without_improvement: usize,
    patience: Option<usize>,
) -> (bool, usize, f32) {
    if current_valid_loss < best_valid_loss {
        // Improvement: reset counter and update best
        (false, 0, current_valid_loss)
    } else {
        // No improvement: increment counter
        let new_count = epochs_without_improvement + 1;
        let should_stop = patience.map_or(false, |p| new_count >= p);
        (should_stop, new_count, best_valid_loss)
    }
}

/// Configure optimizer with training settings from TrainConfig.
/// Applies weight decay and gradient clipping if specified.
fn configure_optimizer_with_training_settings(train_config: &TrainConfig) -> AdamConfig {
    let mut optim = AdamConfig::new();

    if let Some(wd) = train_config.weight_decay {
        optim = optim.with_weight_decay(Some(WeightDecayConfig::new(wd)));
    }
    if let Some(gc) = train_config.grad_clip {
        optim = optim.with_grad_clipping(Some(GradientClippingConfig::Norm(gc)));
    }

    optim
}

/// Persist trained model and config to disk if save path is provided.
/// Model is saved using CompactRecorder; config is saved as JSON alongside.
/// If a save path was provided, both model and config must be saved successfully.
/// Returns Ok(()) if no save was requested, or if both saves succeeded.
/// Returns Err if save was requested but failed for model or config.
fn persist_trained_model_and_config_if_requested<B, M>(
    model: &M,
    model_save_path: &Option<String>,
    train_config: &TrainConfig,
) -> anyhow::Result<()>
where
    B: burn::tensor::backend::Backend,
    M: Clone + burn::module::Module<B>,
{
    if let Some(path) = model_save_path {
        model
            .clone()
            .save_file(path, &CompactRecorder::new())
            .map_err(|e| anyhow::anyhow!("Failed to save model to {}: {}", path, e))?;

        let config_path = format!("{}.config.json", path);
        train_config
            .save(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to save config to {}: {}", config_path, e))?;
    }
    Ok(())
}
pub fn train_sync<B, M>(
    device: &B::Device,
    dataset_training: &impl Dataset<SequenceDatasetItem>,
    dataset_validation: &impl Dataset<SequenceDatasetItem>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    mut model: M,
    model_save_path: Option<String>,
    train_config: TrainConfig,
) -> (M, f32)
where
    B: AutodiffBackend,
    M: SequenceModel<B> + AutodiffModule<B>,
    M::InnerModule: SequenceModel<B::InnerBackend>,
{
    // Initialize optimizer with training settings from config
    let optim_config = configure_optimizer_with_training_settings(&train_config);
    let mut optimizer = optim_config.init();

    // Create batchers
    let batcher_train = SequenceBatcher::<B>::new();
    let batcher_valid = SequenceBatcher::<B::InnerBackend>::new();

    let dataset_train_len = dataset_training.len();
    let dataset_valid_len = dataset_validation.len();

    // Early stopping variables
    let mut best_valid_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0;

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

        let avg_train_loss = compute_average_loss(total_train_loss, num_train_batches);

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

        let avg_valid_loss = compute_average_loss(total_valid_loss, num_valid_batches);

        // ============ Early Stopping Check ============
        let (should_stop, new_epochs_count, new_best_loss) = update_early_stopping_state(
            best_valid_loss,
            avg_valid_loss,
            epochs_without_improvement,
            train_config.patience,
        );
        best_valid_loss = new_best_loss;
        epochs_without_improvement = new_epochs_count;

        if should_stop {
            tracing::info!(
                epoch = epoch + 1,
                best_valid_loss = best_valid_loss,
                "Early stopping triggered",
            );
            break;
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

    // Persist trained model and config if a save path was provided
    persist_trained_model_and_config_if_requested::<B, M>(&model, &model_save_path, &train_config)
        .expect("Failed to persist trained model and config");

    (model, best_valid_loss)
}
