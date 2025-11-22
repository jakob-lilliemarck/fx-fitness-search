use burn::data::dataset::Dataset;

use crate::core::dataset::Metadata;
use std::collections::{HashMap, VecDeque};

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
}

/// Cast from string value to Option<f32> where None represents missing values
pub trait Cast: Send {
    fn cast(&self, input: &HashMap<String, String>) -> Result<Option<f32>, Error>;
    fn clone_box(&self) -> Box<dyn Cast>;
}

/// Interpolate missing values
pub trait Interpolate: Send {
    fn interpolate(&mut self, input: Option<f32>) -> Result<f32, Error>;
    fn clone_box(&self) -> Box<dyn Interpolate>;
}

/// Process an f32 value
pub trait Process: Send {
    fn process(&mut self, input: f32) -> Option<f32>;
    fn clone_box(&self) -> Box<dyn Process>;
}

pub struct Pipeline {
    caster: Box<dyn Cast>,
    interpolator: Box<dyn Interpolate>,
    processors: Vec<Box<dyn Process>>,
    destination: String,
}

impl Clone for Pipeline {
    fn clone(&self) -> Self {
        Self {
            caster: self.caster.clone_box(),
            interpolator: self.interpolator.clone_box(),
            processors: self.processors.iter().map(|p| p.clone_box()).collect(),
            destination: self.destination.clone(),
        }
    }
}

pub trait Ingestable {
    fn ingest(&mut self) -> Result<Sequence, Error>;
    fn headers(&self) -> Vec<String>;
}

pub struct Csv {
    path: String,
    feature_pipelines: Vec<Pipeline>,
    target_pipelines: Vec<Pipeline>,
}

impl Csv {
    pub fn new(
        path: String,
        feature_pipelines: Vec<Pipeline>,
        target_pipelines: Vec<Pipeline>,
    ) -> Self {
        Csv {
            path,
            feature_pipelines,
            target_pipelines,
        }
    }

    fn process_row(
        p: &mut [Pipeline],
        r: &HashMap<String, String>,
    ) -> Result<Option<Vec<f32>>, Error> {
        let mut row = Vec::with_capacity(p.len());
        let mut has_none = false;
        for p in p.iter_mut() {
            let casted = p.caster.cast(&r)?;
            let interpolated = p.interpolator.interpolate(casted)?;

            let mut processed = Some(interpolated);
            for processor in &mut p.processors {
                processed = processed.and_then(|v| processor.process(v));
            }

            match processed {
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
    fn headers(&self) -> Vec<String> {
        self.feature_pipelines
            .iter()
            .map(|p| p.destination.to_string())
            .collect::<Vec<String>>()
    }

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

            let features = Self::process_row(&mut self.feature_pipelines, &record)?;

            let targets = Self::process_row(&mut self.target_pipelines, &record)?;

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

    pub fn get_metadata(
        &self,
        index: usize,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Option<Metadata> {
        let (index, until_idx, target_idx) =
            self.in_bounds(index, sequence_length, prediction_horizon)?;

        let target = self.targets[target_idx].clone();

        Some(Metadata {
            sequence_start_row_id: index.to_string(),
            sequence_end_row_id: until_idx.to_string(),
            target_row_id: target_idx.to_string(),
            target_at_sequence_end: target,
        })
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

pub struct ManySequencesAdapter {
    many_sequences: ManySequences,
    prediction_horizon: usize,
    sequence_length: usize,
}

impl Dataset<(Vec<Vec<f32>>, Vec<f32>)> for ManySequencesAdapter {
    fn get(&self, index: usize) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        let (features, target) =
            self.many_sequences
                .get_item(index, self.sequence_length, self.prediction_horizon)?;
        Some((features.to_vec(), target.clone()))
    }

    fn len(&self) -> usize {
        self.many_sequences
            .len(self.sequence_length, self.prediction_horizon)
    }
}

#[derive(Clone)]
pub struct LinearInterpolator {
    window_size: usize,
    prev_values: VecDeque<Option<f32>>,
}

impl LinearInterpolator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            prev_values: VecDeque::with_capacity(window_size),
        }
    }
}

#[derive(Clone)]
pub struct ErrorInterpolator;

impl Interpolate for ErrorInterpolator {
    fn interpolate(&mut self, input: Option<f32>) -> Result<f32, Error> {
        input.ok_or_else(|| Error::NotEnoughHistory("Missing value not allowed".into()))
    }

    fn clone_box(&self) -> Box<dyn Interpolate> {
        Box::new(self.clone())
    }
}

impl Interpolate for LinearInterpolator {
    fn interpolate(&mut self, input: Option<f32>) -> Result<f32, Error> {
        if let Some(value) = input {
            self.prev_values.push_back(Some(value));
            if self.prev_values.len() > self.window_size {
                self.prev_values.pop_front();
            }
            return Ok(value);
        }

        if self.prev_values.len() < 2 {
            return Err(Error::NotEnoughHistory(
                "Not enough history for interpolation".into(),
            ));
        }

        let values: Vec<f32> = self.prev_values.iter().filter_map(|v| *v).collect();

        if values.len() < 2 {
            return Err(Error::NotEnoughHistory(
                "Cannot interpolate with insufficient non-None values".into(),
            ));
        }

        // Fit line: y = mx + b
        let n = values.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f32>() / n;

        let slope = values
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
            .sum::<f32>()
            / values
                .iter()
                .enumerate()
                .map(|(i, _)| (i as f32 - x_mean).powi(2))
                .sum::<f32>();

        let intercept = y_mean - slope * x_mean;
        let next_x = values.len() as f32;

        Ok(slope * next_x + intercept)
    }

    fn clone_box(&self) -> Box<dyn Interpolate> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, NaiveDate};
    use std::fs::File;
    use std::io::Write;

    #[derive(Clone)]
    struct DayOfWeekCaster;

    impl Cast for DayOfWeekCaster {
        fn cast(&self, input: &HashMap<String, String>) -> Result<Option<f32>, Error> {
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

            let date = NaiveDate::from_ymd_opt(year, month, day)
                .ok_or_else(|| Error::NotEnoughHistory("Invalid date".into()))?;

            let dow = date.weekday().number_from_monday() as f32 - 1.0;
            Ok(Some(dow))
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

        let feature_pipeline = Pipeline {
            caster: Box::new(DayOfWeekCaster),
            interpolator: Box::new(ErrorInterpolator),
            processors: vec![],
            destination: "day_of_week".to_string(),
        };

        let target_pipeline = Pipeline {
            caster: Box::new(DayOfWeekCaster),
            interpolator: Box::new(ErrorInterpolator),
            processors: vec![],
            destination: "target".to_string(),
        };

        let mut csv = Csv {
            path: csv_path.to_string(),
            feature_pipelines: vec![feature_pipeline],
            target_pipelines: vec![target_pipeline],
        };

        let sequence = csv.ingest().unwrap();
        assert_eq!(csv.headers(), vec!["day_of_week"]);
        assert_eq!(sequence.count, 3);

        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0][0], 2.0); // Wednesday (day 1 in 0-indexed)
        assert_eq!(target[0], 3.0); // Thursday
    }
}
