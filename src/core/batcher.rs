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

        // Pre-allocate with capacity
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray;

    #[test]
    fn test_batcher_shapes() {
        let batcher = SequenceBatcher::<B>::new();
        let device = Default::default();

        let items = vec![
            SequenceDatasetItem {
                features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                target: vec![5.0, 6.0],
            },
            SequenceDatasetItem {
                features: vec![vec![7.0, 8.0], vec![9.0, 10.0]],
                target: vec![11.0, 12.0],
            },
        ];

        let batch = batcher.batch(items, &device);

        assert_eq!(batch.sequences.dims(), [2, 2, 2]);
        assert_eq!(batch.targets.dims(), [2, 2]);
    }

    #[test]
    fn test_batcher_data_integrity() {
        let batcher = SequenceBatcher::<B>::new();
        let device = Default::default();

        let items = vec![SequenceDatasetItem {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            target: vec![5.0, 6.0],
        }];

        let batch = batcher.batch(items, &device);

        // Verify shapes
        assert_eq!(batch.sequences.dims(), [1, 2, 2]);
        assert_eq!(batch.targets.dims(), [1, 2]);

        // Verify data by cloning to ndarray and checking values
        let seq_clone = batch.sequences.clone();
        let tgt_clone = batch.targets.clone();

        let expected_sequences = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        );
        let expected_targets =
            Tensor::<B, 2>::from_data(TensorData::new(vec![5.0, 6.0], [1, 2]), &device);

        // Compare using == operator (works for NdArray backend)
        assert_eq!(
            seq_clone.to_data().bytes,
            expected_sequences.to_data().bytes
        );
        assert_eq!(tgt_clone.to_data().bytes, expected_targets.to_data().bytes);
    }

    #[test]
    #[should_panic(expected = "Cannot create a batch from an empty Vec")]
    fn test_batcher_empty_panics() {
        let batcher = SequenceBatcher::<B>::new();
        let device = Default::default();
        batcher.batch(vec![], &device);
    }
}
