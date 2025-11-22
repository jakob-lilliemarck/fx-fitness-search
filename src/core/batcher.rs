use super::dataset::SequenceDatasetItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

/// A batch of training data, ready for the model.
///
/// The batcher's responsibility is to convert a Vec of items (training examples)
/// into stacked tensors that can be passed to the model.
#[derive(Clone, Debug)]
pub struct SequenceBatch<B: Backend> {
    /// Stacked sequences: shape [batch_size, seq_len, features]
    pub sequences: Tensor<B, 3>,
    /// Stacked targets: shape [batch_size, output_size]
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct SequenceBatcher<B: Backend> {
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> SequenceBatcher<B> {
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, SequenceDatasetItem, SequenceBatch<B>> for SequenceBatcher<B> {
    /// Convert a Vec of training items into stacked tensors.
    ///
    /// Input: Vec of (sequence, target) pairs
    /// Output: SequenceBatch with sequences and targets as 3D and 2D tensors respectively
    fn batch(&self, items: Vec<SequenceDatasetItem>, device: &B::Device) -> SequenceBatch<B> {
        let batch_size = items.len();
        assert!(batch_size > 0, "Cannot create a batch from an empty Vec");

        let seq_len = items[0].features.len();
        let feature_dim = items[0].features[0].len();
        let target_dim = items[0].target.len();

        // Pre-allocate with exact capacity - this is the key optimization!
        // No reallocations = much faster and no memory fragmentation
        let total_seq_elements = batch_size * seq_len * feature_dim;
        let total_target_elements = batch_size * target_dim;

        let mut all_sequences = Vec::with_capacity(total_seq_elements);
        let mut all_targets = Vec::with_capacity(total_target_elements);

        // Now extend_from_slice won't cause any reallocations
        for item in items.iter() {
            for timestep in item.features.iter() {
                all_sequences.extend_from_slice(timestep);
            }
            all_targets.extend_from_slice(&item.target);
        }

        // Create tensors directly from the Vec data
        let sequences = Tensor::<B, 3>::from_data(
            TensorData::new(all_sequences, [batch_size, seq_len, feature_dim]),
            device,
        );

        let targets = Tensor::<B, 2>::from_data(
            TensorData::new(all_targets, [batch_size, target_dim]),
            device,
        );

        SequenceBatch { sequences, targets }
    }
}

#[derive(Clone, Debug)]
pub struct MyBatcher<B: Backend> {
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> MyBatcher<B> {
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, (&[Vec<f32>], &Vec<f32>), SequenceBatch<B>> for MyBatcher<B> {
    /// Convert a Vec of training items into stacked tensors.
    ///
    /// Input: Vec of (sequence, target) pairs
    /// Output: SequenceBatch with sequences and targets as 3D and 2D tensors respectively
    fn batch(&self, items: Vec<(&[Vec<f32>], &Vec<f32>)>, device: &B::Device) -> SequenceBatch<B> {
        let batch_size = items.len();
        assert!(batch_size > 0, "Cannot create a batch from an empty Vec");

        let seq_len = items[0].0.len();
        let feature_dim = items[0].0[0].len();
        let target_dim = items[0].1.len();

        // Pre-allocate with exact capacity - this is the key optimization!
        // No reallocations = much faster and no memory fragmentation
        let total_seq_elements = batch_size * seq_len * feature_dim;
        let total_target_elements = batch_size * target_dim;

        let mut all_sequences = Vec::with_capacity(total_seq_elements);
        let mut all_targets = Vec::with_capacity(total_target_elements);

        // Now extend_from_slice won't cause any reallocations
        for (features, target) in items.iter() {
            for timestep in features.iter() {
                all_sequences.extend_from_slice(timestep);
            }
            all_targets.extend_from_slice(target);
        }

        // Create tensors directly from the Vec data
        let sequences = Tensor::<B, 3>::from_data(
            TensorData::new(all_sequences, [batch_size, seq_len, feature_dim]),
            device,
        );

        let targets = Tensor::<B, 2>::from_data(
            TensorData::new(all_targets, [batch_size, target_dim]),
            device,
        );

        SequenceBatch { sequences, targets }
    }
}
