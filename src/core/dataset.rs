use crate::core::ingestion::ManySequences;
use burn::data::dataset::Dataset;

pub type Timestep = Vec<f32>;
pub type RecordID = String;
#[allow(dead_code)]
pub type ColumnID = String;

/// Metadata about the source rows for a sequence item
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Metadata {
    pub sequence_start_row_id: String, // CSV row identifier of first sequence timestep
    pub sequence_end_row_id: String,   // CSV row identifier of last sequence timestep (anchor)
    pub target_row_id: String,         // CSV row identifier of the target value being predicted
    pub target_at_sequence_end: Vec<f32>, // Preprocessed target value at the end of sequence (for naive baseline)
}

pub struct ManySequencesAdapter {
    many_sequences: ManySequences,
    prediction_horizon: usize,
    sequence_length: usize,
}

impl ManySequencesAdapter {
    pub fn new(
        many_sequences: ManySequences,
        prediction_horizon: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            many_sequences,
            prediction_horizon,
            sequence_length,
        }
    }
}

#[derive(Clone)]
pub struct SequenceDatasetItem {
    pub features: Vec<Timestep>,
    pub target: Timestep,
}

impl Dataset<SequenceDatasetItem> for ManySequencesAdapter {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        let (features, target) =
            self.many_sequences
                .get_item(index, self.sequence_length, self.prediction_horizon)?;

        Some(SequenceDatasetItem {
            features: features.to_vec(),
            target: target.clone(),
        })
    }

    fn len(&self) -> usize {
        self.many_sequences
            .len(self.sequence_length, self.prediction_horizon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_many_sequences() -> ManySequences {
        // Single sequence with 5 items
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ];
        let targets = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0], vec![50.0]];
        let metadata = vec![HashMap::new(); 5];

        let sequence = crate::core::ingestion::Sequence::new(features, targets, metadata).unwrap();
        ManySequences::new(vec![sequence])
    }

    #[test]
    fn test_adapter_len() {
        let many_seq = create_test_many_sequences();
        let adapter = ManySequencesAdapter::new(many_seq, 0, 1);

        // With sequence_length=1 and prediction_horizon=0: 5 - 1 - 0 + 1 = 5
        assert_eq!(adapter.len(), 5);
    }

    #[test]
    fn test_adapter_len_with_horizon() {
        let many_seq = create_test_many_sequences();
        let adapter = ManySequencesAdapter::new(many_seq, 1, 2);

        // With sequence_length=2 and prediction_horizon=1: 5 - 2 - 1 + 1 = 3
        assert_eq!(adapter.len(), 3);
    }

    #[test]
    fn test_adapter_get_different_indices() {
        let many_seq = create_test_many_sequences();
        let adapter = ManySequencesAdapter::new(many_seq, 0, 1);

        let item0 = adapter.get(0).unwrap();
        assert_eq!(item0.features[0], vec![1.0, 2.0]);
        assert_eq!(item0.target, vec![10.0]);

        let item2 = adapter.get(2).unwrap();
        assert_eq!(item2.features[0], vec![5.0, 6.0]);
        assert_eq!(item2.target, vec![30.0]);
    }

    #[test]
    fn test_adapter_get_with_sequence_length() {
        let many_seq = create_test_many_sequences();
        let adapter = ManySequencesAdapter::new(many_seq, 0, 2);

        let item = adapter.get(0).unwrap();

        // With sequence_length=2, should get 2 consecutive timesteps
        assert_eq!(item.features.len(), 2);
        assert_eq!(item.features[0], vec![1.0, 2.0]);
        assert_eq!(item.features[1], vec![3.0, 4.0]);
        // Target corresponds to the last feature timestep
        assert_eq!(item.target, vec![20.0]);
    }

    #[test]
    fn test_adapter_get_out_of_bounds() {
        let many_seq = create_test_many_sequences();
        let adapter = ManySequencesAdapter::new(many_seq, 0, 1);

        // With len=5, index 5 should be out of bounds
        assert!(adapter.get(5).is_none());
        assert!(adapter.get(10).is_none());
    }
}
