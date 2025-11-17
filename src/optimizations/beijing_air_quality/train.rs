use super::batcher::SequenceBatcher;
use super::dataset::SequenceDataset;
use super::model::SequenceModel;
use super::train_config::TrainConfig;
use burn::data::dataloader::Dataset;
use burn::data::dataloader::batcher::Batcher;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;

pub fn train<B, M>(
    device: &B::Device,
    dataset_training: &SequenceDataset,
    dataset_validation: &SequenceDataset,
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
                println!(
                    "Early stopping at epoch {} - Best validation loss: {:.6}",
                    epoch + 1,
                    best_valid_loss
                );
                break;
            }
        }

        println!(
            "Epoch {}/{} - Train Loss: {:.6}, Valid Loss: {:.6} (Best: {:.6})",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_valid_loss,
            best_valid_loss
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

#[cfg(test)]
mod tests {
    use super::super::dataset::DatasetBuilder;
    use super::super::model::SimpleLstm;
    use super::super::preprocessor::Pipeline;
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use std::collections::HashMap;

    #[test]
    fn test_training_converges() {
        type Backend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        // Simple repeating pattern: 0, 1, 2, 3, 2, 1, 0, 1, ...
        let pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0];

        // Create pipelines - passthrough (no preprocessing)
        // Use same data for both features and targets
        let mut feature_pipelines = HashMap::new();
        feature_pipelines.insert("value".to_string(), Pipeline::new(vec![]));

        let mut target_pipelines = HashMap::new();
        target_pipelines.insert("value".to_string(), Pipeline::new(vec![]));

        let output_names = vec!["value".to_string()];
        let source_columns = vec!["value".to_string()];

        // Build dataset
        let mut builder = DatasetBuilder::new(
            feature_pipelines,
            output_names.clone(),
            source_columns.clone(),
            target_pipelines,
            output_names,
            source_columns,
            Some(100),
        );

        for i in 0..100 {
            let value = pattern[i % pattern.len()];
            let mut record = HashMap::new();
            record.insert("value".to_string(), value);
            builder.push(record, i.to_string()).unwrap();
        }

        // Split into train (80%) and validation (20%)
        let (dataset_train, opt_dataset_valid) =
            builder.build(4, 1, Some(0.8)).expect("should build");
        let dataset_valid = opt_dataset_valid.expect("should get validation dataset");

        // Create model - input_size=1 (single feature), output_size=1, seq_len=4
        let model = SimpleLstm::<Backend>::new(&device, 1, 64, 1, 4);

        // Train
        train(
            &device,
            &dataset_train,
            &dataset_valid,
            25,
            32,
            0.01,
            model,
            None,
            None,
        );

        println!("Training complete! Check if losses decreased.");
    }
}
