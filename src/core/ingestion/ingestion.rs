use crate::core::{interpolation::Interpolation, preprocessor::Node};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Could not read file: {0}")]
    ReadError(#[from] csv::Error),
    #[error("Interpolation did not have enough hisotry {0}")]
    NotEnoughHistory(String),
    #[error("Encountered None value after saturation")]
    NoneAfterSaturation,
    #[error("Factor out of bounds")]
    FactorOutOfBounds,
    #[error("CastingError: {0}")]
    CastingError(Box<dyn std::error::Error + Send + Sync>),
    #[error("InterpolationError: {0}")]
    InterpolationError(Box<dyn std::error::Error + Send + Sync>),
    #[error("ProcessingError: {0}")]
    ProcessingError(Box<dyn std::error::Error + Send + Sync>),
    #[error("SequenceError: {0}")]
    SequenceError(#[from] SequenceError),
}

/// Cast from string value to Option<f32> where None represents missing values
pub trait Cast: Send {
    fn cast(&self, input: &HashMap<String, String>) -> Result<HashMap<String, f32>, Error>;
    fn clone_box(&self) -> Box<dyn Cast>;
}

// Transforms are processed in order
// Values occupy the position at which point they were tranformed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extract {
    pub source: String,
    pub interpolation: Interpolation,
    pub nodes: Vec<Node>,
}

impl Extract {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
            interpolation: Interpolation::Error,
            nodes: Vec::new(),
        }
    }

    pub fn with_node(mut self, n: Node) -> Self {
        self.nodes.push(n);
        self
    }

    pub fn with_interpolation(mut self, i: Interpolation) -> Self {
        self.interpolation = i;
        self
    }
}

pub trait Ingestable {
    fn ingest(&mut self) -> Result<Sequence, Error>;
}

pub struct Csv {
    path: String,
    cast: Box<dyn Cast>,
    feature_pipelines: Vec<Extract>,
    target_pipelines: Vec<Extract>,
}

impl Csv {
    pub fn new(
        path: String,
        cast: Box<dyn Cast>,
        feature_pipelines: Vec<Extract>,
        target_pipelines: Vec<Extract>,
    ) -> Self {
        Csv {
            path,
            cast,
            feature_pipelines,
            target_pipelines,
        }
    }

    fn process_row(t: &mut [Extract], r: &HashMap<String, f32>) -> Result<Option<Vec<f32>>, Error> {
        let mut row = Vec::with_capacity(t.len());
        let mut has_none = false;
        for t in t.iter_mut() {
            // For Noop features (disabled features), output 0.0 to maintain fixed vector size
            if t.source == "Noop" {
                row.push(0.0);
                continue;
            }

            let input = r.get(&t.source).map(Clone::clone);

            let mut output = t.interpolation.interpolate(input).map_err(|e| {
                Error::InterpolationError(
                    format!("Failed to interpolate '{}': {}", t.source, e).into(),
                )
            })?;

            for processor in &mut t.nodes {
                output = output.and_then(|v| processor.process(v));
            }

            // Only push if there is a value
            match output {
                Some(value) => {
                    if !has_none {
                        row.push(value)
                    }
                }
                None => has_none = true,
            }
        }

        if has_none { Ok(None) } else { Ok(Some(row)) }
    }
}

impl Ingestable for Csv {
    fn ingest(&mut self) -> Result<Sequence, Error> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&self.path)?;

        let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

        let mut seen_complete = false;
        let mut buf_features = Vec::new();
        let mut buf_targets = Vec::new();
        let mut buf_metadata = Vec::new();

        while let Some(record) = reader.records().next() {
            let record: HashMap<String, String> = headers
                .iter()
                .zip(record?.iter())
                .map(|(h, v)| (h.to_string(), v.to_string()))
                .collect();

            let casted = self.cast.cast(&record)?;

            let features = Self::process_row(&mut self.feature_pipelines, &casted)?;

            let targets = Self::process_row(&mut self.target_pipelines, &casted)?;

            match (features, targets, seen_complete) {
                (Some(features), Some(targets), _) => {
                    if !seen_complete {
                        seen_complete = true
                    }
                    buf_features.push(features);
                    buf_targets.push(targets);
                    buf_metadata.push(record)
                }
                (_, _, false) => continue,
                (_, _, true) => {
                    return Err(Error::NoneAfterSaturation);
                }
            }
        }

        let sequence = Sequence::new(buf_features, buf_targets, buf_metadata)?;
        Ok(sequence)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Could not create Sequence:")]
pub struct SequenceError(String);

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
    ) -> Result<Self, SequenceError> {
        let count = features.len();
        if targets.len() != count {
            return Err(SequenceError(format!(
                "targets length ({}) must match features length ({})",
                targets.len(),
                count
            )));
        }
        if metadata.len() != count {
            return Err(SequenceError(format!(
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
            return Err(Error::FactorOutOfBounds);
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

    // the number of complete items in this sequence
    pub fn len(&self, sequence_length: usize, prediction_horizon: usize) -> usize {
        self.count
            .saturating_sub(sequence_length + prediction_horizon - 1)
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
            return Err(Error::FactorOutOfBounds);
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, NaiveDate};
    use std::fs::File;
    use std::io::Write;

    #[derive(Clone)]
    struct ExampleCast;

    impl Cast for ExampleCast {
        fn cast(&self, input: &HashMap<String, String>) -> Result<HashMap<String, f32>, Error> {
            let mut cast = HashMap::with_capacity(2);

            let year = input
                .get("Year")
                .and_then(|y| y.parse::<i32>().ok())
                .ok_or_else(|| Error::NotEnoughHistory("Missing or invalid Year".into()))?;

            let month = input
                .get("Month")
                .and_then(|m| m.parse::<u32>().ok())
                .ok_or_else(|| Error::NotEnoughHistory("Missing or invalid Month".into()))?;

            let day = input
                .get("Day")
                .and_then(|d| d.parse::<u32>().ok())
                .ok_or_else(|| Error::NotEnoughHistory("Missing or invalid Day".into()))?;

            let dow = NaiveDate::from_ymd_opt(year, month, day)
                .map(|d| (d.weekday().number_from_monday() - 1) as f32)
                .ok_or_else(|| Error::NotEnoughHistory("Invalid date".into()))?;

            cast.insert("dow".to_string(), dow);

            if let Some(value) = input.get("Value") {
                let trimmed = value.trim();
                // Treat "NA" as missing (don't insert into cast)
                if !trimmed.eq_ignore_ascii_case("NA") {
                    let v = trimmed
                        .parse::<f32>()
                        .map_err(|err| Error::CastingError(Box::new(err)))?;
                    cast.insert("Value".to_string(), v);
                }
            }

            Ok(cast)
        }

        fn clone_box(&self) -> Box<dyn Cast> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_csv_ingestion_with_day_of_week() {
        let csv_path = "/tmp/test_ingest.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        writeln!(file, "2025,1,15,5.5").unwrap();
        writeln!(file, "2025,1,16,6.2").unwrap();
        writeln!(file, "2025,1,17,7.1").unwrap();

        let mut csv = Csv {
            path: csv_path.to_string(),
            cast: Box::new(ExampleCast),
            feature_pipelines: vec![Extract::new("dow")],
            target_pipelines: vec![Extract::new("dow")],
        };

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count, 3);

        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        assert_eq!(features.len(), 1);
        // First row: 2025-01-15 is Wednesday (dow=2)
        assert_eq!(features[0][0], 2.0);
        // Target with prediction_horizon=0 is at the same position: Wednesday (dow=2)
        assert_eq!(target[0], 2.0);
    }

    #[test]
    fn test_multiple_features_and_targets() {
        // Behavior: multiple Extract pipelines should produce multiple columns in features and targets.
        let csv_path = "/tmp/test_ingest_multi.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        writeln!(file, "2025,1,15,5.5").unwrap();
        writeln!(file, "2025,1,16,6.2").unwrap();

        let mut csv = Csv {
            path: csv_path.to_string(),
            cast: Box::new(ExampleCast),
            feature_pipelines: vec![Extract::new("dow"), Extract::new("Value")],
            target_pipelines: vec![Extract::new("Value"), Extract::new("dow")],
        };

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count, 2);

        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        // Two features: dow and Value
        assert_eq!(features[0].len(), 2);
        assert_eq!(features[0][0], 2.0); // dow for Wednesday
        assert_eq!(features[0][1], 5.5); // Value
        // Two targets: Value and dow
        assert_eq!(target.len(), 2);
        assert_eq!(target[0], 5.5); // Value
        assert_eq!(target[1], 2.0); // dow
    }

    #[test]
    fn test_prediction_horizon_offsets_target() {
        // Behavior: with prediction_horizon > 0, the target should be offset to a future row.
        // prediction_horizon=1 means target is 1 row ahead.
        let csv_path = "/tmp/test_ingest_horizon.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        writeln!(file, "2025,1,15,5.5").unwrap();
        writeln!(file, "2025,1,16,6.2").unwrap();
        writeln!(file, "2025,1,17,7.1").unwrap();

        let mut csv = Csv {
            path: csv_path.to_string(),
            cast: Box::new(ExampleCast),
            feature_pipelines: vec![Extract::new("Value")],
            target_pipelines: vec![Extract::new("Value")],
        };

        let sequence = csv.ingest().unwrap();
        // With sequence_length=1 and prediction_horizon=1, we can have 1 complete item (3 - 1 - 1 = 1)
        let (features, target) = sequence.get_item(0, 1, 1).unwrap();
        // Features: row 0 (Value=5.5)
        assert_eq!(features[0][0], 5.5);
        // Target: row 1 (Value=6.2) because prediction_horizon=1 offsets by 1
        assert_eq!(target[0], 6.2);
    }

    #[test]
    fn test_interpolation_fills_gaps() {
        // Behavior: with linear interpolation, gaps in data can be filled using history.
        use crate::core::interpolation::{Interpolation, LinearInterpolator};

        let csv_path = "/tmp/test_ingest_interp.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        // Values form a linear pattern: 10, 20, NA -> should interpolate to 30
        writeln!(file, "2025,1,15,10.0").unwrap();
        writeln!(file, "2025,1,16,20.0").unwrap();
        writeln!(file, "2025,1,17,NA").unwrap();

        let mut csv = Csv {
            path: csv_path.to_string(),
            cast: Box::new(ExampleCast),
            feature_pipelines: vec![
                Extract::new("Value")
                    .with_interpolation(Interpolation::Linear(LinearInterpolator::new(2))),
            ],
            target_pipelines: vec![Extract::new("dow")],
        };

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count, 3);
        // Third row should be interpolated to 30.0
        let (features, _target) = sequence.get_item(2, 1, 0).unwrap();
        assert!((features[0][0] - 30.0).abs() < 1e-4);
    }
}

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
        assert_eq!(train.count, 6);
        assert_eq!(validation.count, 4);
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
        assert_eq!(train.count, 0);
        assert_eq!(validation.count, 5);
    }

    #[test]
    fn split_factor_one_produces_empty_validation() {
        // Behavior: split(1.0) produces train with all items and empty validation.
        let sequence = create_test_sequence(5);

        let (train, validation) = sequence.split(1.0).unwrap();
        assert_eq!(train.count, 5);
        assert_eq!(validation.count, 0);
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
