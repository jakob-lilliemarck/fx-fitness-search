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

        let mut count: usize = 0;
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
                    count += 1;
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

        Ok(Sequence {
            count,
            features: buf_features,
            targets: buf_targets,
            metadata: buf_metadata,
        })
    }
}

pub struct Sequence {
    count: usize,
    features: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    metadata: Vec<HashMap<String, String>>,
}

impl Sequence {
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

        let train = Sequence {
            count: split_idx,
            features: self.features[..split_idx].to_vec(),
            targets: self.targets[..split_idx].to_vec(),
            metadata: self.metadata[..split_idx].to_vec(),
        };

        let validation = Sequence {
            count: self.count - split_idx,
            features: self.features[split_idx..].to_vec(),
            targets: self.targets[split_idx..].to_vec(),
            metadata: self.metadata[split_idx..].to_vec(),
        };

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
    use crate::core::preprocessor::{Cos, Node, Sin};

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
            feature_pipelines: vec![
                Extract::new("dow").with_node(Node::Cos(Cos::new(7.0))),
                Extract::new("dow").with_node(Node::Sin(Sin::new(7.0))),
            ],
            target_pipelines: vec![Extract::new("dow")],
        };

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count, 3);

        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0][0], 2.0); // Wednesday (day 1 in 0-indexed)
        assert_eq!(target[0], 3.0); // Thursday
    }
}
