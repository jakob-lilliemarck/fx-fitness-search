use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Split factor must be between 0.0 and 1.0, got: {0}")]
    SplitFactor(f32),
    #[error("ValidationError: {0}")]
    Validation(String),
}

#[derive(Debug)]
pub struct Sequence {
    count: usize,
    features: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    metadata: Vec<HashMap<String, String>>,
}

impl Sequence {
    pub fn new(
        features: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        metadata: Vec<HashMap<String, String>>,
    ) -> Result<Self, Error> {
        let count = features.len();
        if targets.len() != count {
            return Err(Error::Validation(format!(
                "targets length ({}) must match features length ({})",
                targets.len(),
                count
            )));
        }
        if metadata.len() != count {
            return Err(Error::Validation(format!(
                "metadata length ({}) must match features length ({})",
                metadata.len(),
                count
            )));
        }
        Ok(Self {
            count,
            features,
            targets,
            metadata,
        })
    }

    pub fn get_item(
        &self,
        index: usize,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Option<(&[Vec<f32>], &Vec<f32>)> {
        let (index, until_idx, target_idx) =
            self.in_bounds(index, sequence_length, prediction_horizon)?;

        Some((&self.features[index..=until_idx], &self.targets[target_idx]))
    }

    pub fn split(self, factor: f32) -> Result<(Sequence, Sequence), Error> {
        if factor < 0.0 || factor > 1.0 {
            return Err(Error::SplitFactor(factor));
        }

        let split_idx = ((self.count as f32) * factor) as usize;

        let train = Sequence::new(
            self.features[..split_idx].to_vec(),
            self.targets[..split_idx].to_vec(),
            self.metadata[..split_idx].to_vec(),
        )?;

        let validation = Sequence::new(
            self.features[split_idx..].to_vec(),
            self.targets[split_idx..].to_vec(),
            self.metadata[split_idx..].to_vec(),
        )?;

        Ok((train, validation))
    }

    /// The numer of records in the sequence
    pub fn count(&self) -> usize {
        self.count
    }

    /// the number of complete training items in this sequence
    pub fn len(&self, sequence_length: usize, prediction_horizon: usize) -> usize {
        self.count
            .saturating_sub(sequence_length + prediction_horizon - 1)
    }

    fn in_bounds(
        &self,
        index: usize,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Option<(usize, usize, usize)> {
        if index >= self.len(sequence_length, prediction_horizon) {
            return None;
        }

        let until_idx = self.until(index, sequence_length);
        if until_idx >= self.features.len() {
            return None;
        }

        let target_idx = self.target(index, sequence_length, prediction_horizon);
        if target_idx >= self.targets.len() {
            return None;
        }

        Some((index, until_idx, target_idx))
    }

    /// The index at the end of the range
    fn until(&self, index: usize, sequence_length: usize) -> usize {
        index + sequence_length - 1
    }

    /// The index of the target
    /// With prediction_horizon=0, target is at the same timestep as the last feature.
    /// With prediction_horizon=1, target is 1 step ahead, etc.
    fn target(&self, index: usize, sequence_length: usize, prediction_horizon: usize) -> usize {
        self.until(index, sequence_length) + prediction_horizon
    }
}

pub struct ManySequences {
    sequences: Vec<Sequence>,
}

impl ManySequences {
    pub fn new(sequences: Vec<Sequence>) -> Self {
        Self { sequences }
    }

    fn in_bounds(
        &self,
        index: usize,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Option<(usize, usize)> {
        // (sequence_id, local_index)
        let mut cumulative = 0;
        for (seq_id, seq) in self.sequences.iter().enumerate() {
            let seq_len = seq.len(sequence_length, prediction_horizon);
            if index < cumulative + seq_len {
                return Some((seq_id, index - cumulative));
            }
            cumulative += seq_len;
        }
        None
    }

    pub fn len(&self, sequence_length: usize, prediction_horizon: usize) -> usize {
        self.sequences
            .iter()
            .map(|s| s.len(sequence_length, prediction_horizon))
            .sum()
    }

    pub fn get_item(
        &self,
        index: usize,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Option<(&[Vec<f32>], &Vec<f32>)> {
        let (seq_id, local_idx) = self.in_bounds(index, sequence_length, prediction_horizon)?;
        self.sequences[seq_id].get_item(local_idx, sequence_length, prediction_horizon)
    }

    pub fn split(self, factor: f32) -> Result<(ManySequences, ManySequences), Error> {
        if factor < 0.0 || factor > 1.0 {
            return Err(Error::SplitFactor(factor));
        }
        let (trains, validations): (Vec<_>, Vec<_>) = self
            .sequences
            .into_iter()
            .map(|seq| seq.split(factor))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        Ok((ManySequences::new(trains), ManySequences::new(validations)))
    }
}

// ============================================================
// Sequence tests
// ============================================================
#[cfg(test)]
mod sequence_tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_sequence(count: usize) -> Sequence {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        let mut metadata = Vec::new();

        for i in 0..count {
            features.push(vec![i as f32]);
            targets.push(vec![i as f32]);
            let mut meta = HashMap::new();
            meta.insert("index".to_string(), i.to_string());
            metadata.push(meta);
        }

        Sequence::new(features, targets, metadata).unwrap()
    }

    #[test]
    fn get_item_basic_access() {
        // Behavior: get_item(index, sequence_length, prediction_horizon) retrieves a contiguous
        // window of features and the corresponding target.
        // With prediction_horizon=0, target is at the same timestep as the last feature.
        let sequence = create_test_sequence(5);

        // Get first item: index=0, sequence_length=1, prediction_horizon=0
        // Features: row 0, target: row 0 (same timestep)
        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        assert_eq!(features, &[vec![0.0]]);
        assert_eq!(target, &[0.0]);
    }

    #[test]
    fn get_item_sequence_length_multiple_rows() {
        // Behavior: sequence_length > 1 returns multiple consecutive feature rows as a slice.
        // Target is at the same timestep as the last feature when prediction_horizon=0.
        let sequence = create_test_sequence(5);

        // Get item with sequence_length=2, prediction_horizon=0
        // Features: rows 0-1, target: row 1 (same as last feature)
        let (features, target) = sequence.get_item(0, 2, 0).unwrap();
        assert_eq!(features, &[vec![0.0], vec![1.0]]);
        assert_eq!(target, &[1.0]);
    }

    #[test]
    fn get_item_prediction_horizon_offset() {
        // Behavior: prediction_horizon > 0 offsets the target forward from the last feature.
        // With sequence_length=1 and prediction_horizon=1, target is 1 step ahead.
        let sequence = create_test_sequence(5);

        let (features, target) = sequence.get_item(0, 1, 1).unwrap();
        assert_eq!(features, &[vec![0.0]]);
        // target_idx = (0 + 1 - 1) + 1 = 1, so target is row 1
        assert_eq!(target, &[1.0]);
    }

    #[test]
    fn get_item_boundary_valid_range() {
        // Behavior: get_item returns valid items while within bounds.
        // With count=5, sequence_length=2, prediction_horizon=1:
        // - len(2, 1) = 5 - 2 - 1 + 1 = 3 valid items (indices 0, 1, 2)
        let sequence = create_test_sequence(5);

        // Last valid index should be 2
        assert!(sequence.get_item(2, 2, 1).is_some());
        // Index 3 should be out of bounds
        assert!(sequence.get_item(3, 2, 1).is_none());
    }

    #[test]
    fn get_item_out_of_bounds() {
        // Behavior: get_item returns None when index is out of bounds.
        let sequence = create_test_sequence(5);

        assert!(sequence.get_item(5, 1, 0).is_none());
        assert!(sequence.get_item(10, 1, 0).is_none());
    }

    #[test]
    fn len_with_sequence_and_horizon_parameters() {
        // Behavior: len(sequence_length, prediction_horizon) returns the number of valid items.
        // Formula: count - sequence_length - prediction_horizon + 1
        let sequence = create_test_sequence(10);

        // len(1, 0) = 10 - 1 - 0 + 1 = 10
        assert_eq!(sequence.len(1, 0), 10);

        // len(2, 0) = 10 - 2 - 0 + 1 = 9
        assert_eq!(sequence.len(2, 0), 9);

        // len(1, 1) = 10 - 1 - 1 + 1 = 9
        assert_eq!(sequence.len(1, 1), 9);

        // len(3, 2) = 10 - 3 - 2 + 1 = 6
        assert_eq!(sequence.len(3, 2), 6);
    }

    #[test]
    fn split_creates_two_sequences() {
        // Behavior: split(factor) divides the sequence at factor*count into train and validation.
        // Factor=0.6 on 10 items -> 6 train, 4 validation
        let sequence = create_test_sequence(10);

        let (train, validation) = sequence.split(0.6).unwrap();
        assert_eq!(train.count(), 6);
        assert_eq!(validation.count(), 4);
    }

    #[test]
    fn split_preserves_data_integrity() {
        // Behavior: train and validation sets contain the correct features and targets.
        let sequence = create_test_sequence(5);

        let (train, validation) = sequence.split(0.6).unwrap();
        // train has 3 items (indices 0, 1, 2)
        assert_eq!(train.features, vec![vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(train.targets, vec![vec![0.0], vec![1.0], vec![2.0]]);

        // validation has 2 items (indices 3, 4)
        assert_eq!(validation.features, vec![vec![3.0], vec![4.0]]);
        assert_eq!(validation.targets, vec![vec![3.0], vec![4.0]]);
    }

    #[test]
    fn split_factor_zero_produces_empty_train() {
        // Behavior: split(0.0) produces an empty train set and validation with all items.
        let sequence = create_test_sequence(5);

        let (train, validation) = sequence.split(0.0).unwrap();
        assert_eq!(train.count(), 0);
        assert_eq!(validation.count(), 5);
    }

    #[test]
    fn split_factor_one_produces_empty_validation() {
        // Behavior: split(1.0) produces train with all items and empty validation.
        let sequence = create_test_sequence(5);

        let (train, validation) = sequence.split(1.0).unwrap();
        assert_eq!(train.count(), 5);
        assert_eq!(validation.count(), 0);
    }

    #[test]
    fn split_factor_out_of_bounds_errors() {
        // Behavior: split with factor < 0.0 or > 1.0 returns an error.
        let sequence1 = create_test_sequence(5);
        let sequence2 = create_test_sequence(5);

        assert!(sequence1.split(-0.1).is_err());
        assert!(sequence2.split(1.1).is_err());
    }
}

// ============================================================
// ManySequences tests
// ============================================================
#[cfg(test)]
mod many_sequences_tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_many_sequences() -> ManySequences {
        // Create 3 sequences of different sizes: 3, 2, 4 items
        let sequences = vec![
            Sequence::new(
                vec![vec![0.0], vec![1.0], vec![2.0]],
                vec![vec![0.0], vec![1.0], vec![2.0]],
                vec![HashMap::new(), HashMap::new(), HashMap::new()],
            )
            .unwrap(),
            Sequence::new(
                vec![vec![10.0], vec![11.0]],
                vec![vec![10.0], vec![11.0]],
                vec![HashMap::new(), HashMap::new()],
            )
            .unwrap(),
            Sequence::new(
                vec![vec![20.0], vec![21.0], vec![22.0], vec![23.0]],
                vec![vec![20.0], vec![21.0], vec![22.0], vec![23.0]],
                vec![
                    HashMap::new(),
                    HashMap::new(),
                    HashMap::new(),
                    HashMap::new(),
                ],
            )
            .unwrap(),
        ];

        ManySequences::new(sequences)
    }

    #[test]
    fn get_item_from_first_sequence() {
        // Behavior: get_item maps global index 0 to the first sequence.
        let many = create_test_many_sequences();

        let (features, target) = many.get_item(0, 1, 0).unwrap();
        assert_eq!(features, &[vec![0.0]]);
        assert_eq!(target, &[0.0]);
    }

    #[test]
    fn get_item_from_second_sequence() {
        // Behavior: get_item with index 3 crosses into the second sequence.
        // Sequence 1 has 3 items (indices 0-2), so index 3 maps to second sequence, local index 0.
        let many = create_test_many_sequences();

        let (features, target) = many.get_item(3, 1, 0).unwrap();
        assert_eq!(features, &[vec![10.0]]);
        assert_eq!(target, &[10.0]);
    }

    #[test]
    fn get_item_from_third_sequence() {
        // Behavior: get_item with index 5 crosses into the third sequence.
        // Sequences 1-2 have 3+2=5 items (indices 0-4), so index 5 maps to third sequence, local index 0.
        let many = create_test_many_sequences();

        let (features, target) = many.get_item(5, 1, 0).unwrap();
        assert_eq!(features, &[vec![20.0]]);
        assert_eq!(target, &[20.0]);
    }

    #[test]
    fn get_item_with_sequence_length() {
        // Behavior: sequence_length works correctly when retrieving multiple rows.
        let many = create_test_many_sequences();

        let (features, target) = many.get_item(0, 2, 0).unwrap();
        assert_eq!(features, &[vec![0.0], vec![1.0]]);
        assert_eq!(target, &[1.0]);
    }

    #[test]
    fn get_item_out_of_bounds() {
        // Behavior: get_item returns None when index is out of bounds.
        let many = create_test_many_sequences();

        // Total items with sequence_length=1, prediction_horizon=0: 3+2+4=9
        assert!(many.get_item(9, 1, 0).is_none());
        assert!(many.get_item(100, 1, 0).is_none());
    }

    #[test]
    fn len_aggregates_across_sequences() {
        // Behavior: len(sequence_length, prediction_horizon) sums lengths from all sequences.
        let many = create_test_many_sequences();

        // Each sequence calculates its own len, then sum:
        // Seq1: 3 - 1 - 0 + 1 = 3
        // Seq2: 2 - 1 - 0 + 1 = 2
        // Seq3: 4 - 1 - 0 + 1 = 4
        // Total: 3 + 2 + 4 = 9
        assert_eq!(many.len(1, 0), 9);
    }

    #[test]
    fn len_with_prediction_horizon() {
        // Behavior: len correctly accounts for prediction_horizon in each sequence.
        let many = create_test_many_sequences();

        // With prediction_horizon=1:
        // Seq1: 3 - 1 - 1 + 1 = 2
        // Seq2: 2 - 1 - 1 + 1 = 1
        // Seq3: 4 - 1 - 1 + 1 = 3
        // Total: 2 + 1 + 3 = 6
        assert_eq!(many.len(1, 1), 6);
    }

    #[test]
    fn split_divides_all_sequences() {
        // Behavior: split applies the factor to each sequence independently.
        let many = create_test_many_sequences();

        let (train, validation) = many.split(0.5).unwrap();
        // Each sequence splits at 50%:
        // Seq1: train 1 (0), validation 2 (1,2)
        // Seq2: train 1 (10), validation 1 (11)
        // Seq3: train 2 (20,21), validation 2 (22,23)
        assert_eq!(train.len(1, 0), 1 + 1 + 2);
        assert_eq!(validation.len(1, 0), 2 + 1 + 2);
    }

    #[test]
    fn split_preserves_sequence_count() {
        // Behavior: split maintains the number of sequences in both train and validation.
        let many = create_test_many_sequences();

        let (train, validation) = many.split(0.6).unwrap();
        assert_eq!(train.sequences.len(), 3);
        assert_eq!(validation.sequences.len(), 3);
    }

    #[test]
    fn split_factor_zero() {
        // Behavior: split(0.0) produces empty train and validation has all sequences.
        let many = create_test_many_sequences();

        let (train, validation) = many.split(0.0).unwrap();
        // Train has 0 items in each sequence
        assert_eq!(train.len(1, 0), 0);
        // Validation has all items
        assert_eq!(validation.len(1, 0), 3 + 2 + 4);
    }

    #[test]
    fn split_factor_one() {
        // Behavior: split(1.0) produces train with all sequences and empty validation.
        let many = create_test_many_sequences();

        let (train, validation) = many.split(1.0).unwrap();
        // Train has all items
        assert_eq!(train.len(1, 0), 3 + 2 + 4);
        // Validation has 0 items in each sequence
        assert_eq!(validation.len(1, 0), 0);
    }
}
