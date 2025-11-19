use super::preprocessor::Pipeline;
use burn::data::dataset::Dataset;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

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

pub struct DatasetBuilder {
    feature_pipelines: HashMap<String, Pipeline>,
    feature_output_names: Vec<String>,
    feature_source_columns: Vec<String>,
    feature_cache: HashMap<String, f32>,

    target_pipelines: HashMap<String, Pipeline>,
    target_output_names: Vec<String>,
    target_source_columns: Vec<String>,
    target_cache: HashMap<String, f32>,

    features: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    record_ids: Vec<RecordID>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No pipeline configured for feature \"{0}\"")]
    MissingPipeline(String),
    #[error("Record has no value at key \"{0}\"")]
    MissingValue(String),
    #[error("Split factor must be a number between 0.0 and 1.0, got: {0}")]
    InvalidSplitFactor(f32),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl DatasetBuilder {
    pub fn new(
        feature_pipelines: HashMap<String, Pipeline>,
        feature_output_names: Vec<String>,
        feature_source_columns: Vec<String>,
        target_pipelines: HashMap<String, Pipeline>,
        target_output_names: Vec<String>,
        target_source_columns: Vec<String>,
        size_hint: Option<usize>,
    ) -> Self {
        Self {
            feature_pipelines,
            feature_output_names,
            feature_source_columns,
            feature_cache: HashMap::new(),

            target_pipelines,
            target_output_names,
            target_source_columns,
            target_cache: HashMap::new(),

            features: Vec::with_capacity(size_hint.unwrap_or(0)),
            targets: Vec::with_capacity(size_hint.unwrap_or(0)),
            record_ids: Vec::with_capacity(size_hint.unwrap_or(0)),
        }
    }

    // NOTE!
    // - Required records to be pushed IN ORDER!
    // - Currently does not forward fill
    pub fn push(&mut self, record: HashMap<String, f32>, id: String) -> Result<(), Error> {
        // Process features
        let mut feature_timestep = Vec::with_capacity(self.feature_output_names.len());

        for (output_name, source_column) in self
            .feature_output_names
            .iter()
            .zip(self.feature_source_columns.iter())
        {
            let pipeline = match self.feature_pipelines.get_mut(output_name) {
                Some(pipeline) => pipeline,
                None => return Err(Error::MissingPipeline(output_name.to_string())),
            };

            match record.get(source_column) {
                Some(value) => {
                    if let Some(value) = pipeline.process(value.to_owned()) {
                        self.feature_cache.insert(output_name.to_string(), value);
                        feature_timestep.push(value)
                    }
                }
                None => {
                    if let Some(value) = self.feature_cache.get(output_name) {
                        feature_timestep.push(value.to_owned())
                    } else {
                        return Err(Error::MissingValue(source_column.to_string()));
                    }
                }
            };
        }

        // Process targets
        let mut target_timestep = Vec::with_capacity(self.target_output_names.len());

        for (output_name, source_column) in self
            .target_output_names
            .iter()
            .zip(self.target_source_columns.iter())
        {
            let pipeline = match self.target_pipelines.get_mut(output_name) {
                Some(pipeline) => pipeline,
                None => return Err(Error::MissingPipeline(output_name.to_string())),
            };

            match record.get(source_column) {
                Some(value) => {
                    if let Some(value) = pipeline.process(value.to_owned()) {
                        self.target_cache.insert(output_name.to_string(), value);
                        target_timestep.push(value)
                    }
                }
                None => {
                    if let Some(value) = self.target_cache.get(output_name) {
                        target_timestep.push(value.to_owned())
                    } else {
                        return Err(Error::MissingValue(source_column.to_string()));
                    }
                }
            };
        }

        // Only push if we got all features AND all targets
        if self.feature_output_names.len() == feature_timestep.len()
            && self.target_output_names.len() == target_timestep.len()
        {
            self.features.push(feature_timestep);
            self.targets.push(target_timestep);
            self.record_ids.push(id);
        };
        Ok(())
    }

    pub fn build(
        self,
        sequence_length: usize,
        prediction_horizon: usize,
        split: Option<f32>,
    ) -> Result<(SequenceDataset, Option<SequenceDataset>), Error> {
        match split {
            Some(ratio) => {
                if ratio > 1.0 || ratio < 0.0 {
                    return Err(Error::InvalidSplitFactor(ratio));
                }

                let split_idx = (self.features.len() as f32 * ratio) as usize;

                let (training_features, validation_features) = self.features.split_at(split_idx);
                let (training_targets, validation_targets) = self.targets.split_at(split_idx);
                let (training_ids, validation_ids) = self.record_ids.split_at(split_idx);

                // Pre-materialize all items during dataset creation
                let training = SequenceDataset::from_features_and_targets(
                    training_features.to_vec(),
                    training_targets.to_vec(),
                    training_ids.to_vec(),
                    sequence_length,
                    prediction_horizon,
                );

                let validation = SequenceDataset::from_features_and_targets(
                    validation_features.to_vec(),
                    validation_targets.to_vec(),
                    validation_ids.to_vec(),
                    sequence_length,
                    prediction_horizon,
                );

                Ok((training, Some(validation)))
            }
            None => {
                let dataset = SequenceDataset::from_features_and_targets(
                    self.features,
                    self.targets,
                    self.record_ids,
                    sequence_length,
                    prediction_horizon,
                );
                Ok((dataset, None))
            }
        }
    }

    /// Dump the dataset to a CSV file
    #[allow(dead_code)]
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut file = File::create(path)?;

        // Write header (semicolon-separated for European locale)
        let mut header = self.feature_output_names.clone();
        header.extend(self.target_output_names.clone());
        writeln!(file, "{}", header.join(";"))?;

        // Write each timestep as a row (with comma as decimal separator)
        for (features, targets) in self.features.iter().zip(self.targets.iter()) {
            let mut row: Vec<String> = features
                .iter()
                .map(|v| v.to_string().replace('.', ","))
                .collect();
            row.extend(targets.iter().map(|v| v.to_string().replace('.', ",")));
            writeln!(file, "{}", row.join(";"))?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct SequenceDatasetItem {
    pub sequence: Vec<Timestep>,
    pub target: Timestep,
}

pub struct SequenceDataset {
    items: Vec<(Metadata, SequenceDatasetItem)>,
}

impl SequenceDataset {
    /// Build dataset from separate features and targets by pre-creating all items
    pub fn from_features_and_targets(
        features: Vec<Timestep>,
        targets: Vec<Timestep>,
        record_ids: Vec<RecordID>,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Self {
        let mut items = Vec::new();

        // Pre-create all sequence items once
        for index in 0..features.len() {
            let end_index = index + sequence_length - 1;
            let target_index = end_index + prediction_horizon;

            if target_index >= features.len() || target_index >= targets.len() {
                break;
            }

            let start_index = index;

            // Create Metadata from record_ids and target value at sequence end
            let metadata = Metadata {
                sequence_start_row_id: record_ids[start_index].clone(),
                sequence_end_row_id: record_ids[end_index].clone(),
                target_row_id: record_ids[target_index].clone(),
                target_at_sequence_end: targets[end_index].clone(),
            };

            // Create sequence and target as before
            let sequence = features[index..index + sequence_length].to_vec();
            let target = targets[target_index].clone();

            items.push((metadata, SequenceDatasetItem { sequence, target }));
        }

        Self { items }
    }

    /// Build dataset from pre-created tuples of metadata and items.
    ///
    /// This is useful for combining items from multiple sources (e.g., different
    /// weather stations) where each source creates its own internally-consistent
    /// sequences, but they can be safely mixed for training.
    pub fn from_items(items: Vec<(Metadata, SequenceDatasetItem)>) -> Self {
        Self { items }
    }
}

impl SequenceDataset {
    /// Retrieve metadata for a specific dataset item
    pub fn get_metadata(&self, index: usize) -> Option<&Metadata> {
        self.items.get(index).map(|(metadata, _)| metadata)
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        // Extract the item from the tuple, cloning it
        self.items.get(index).map(|(_, item)| item.clone())
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::preprocessor::Pipeline;
    use super::*;

    /// Helper to build a simple dataset with integers for easy visual verification.
    /// Features: [[1], [2], [3], ..., [num_timesteps]]
    /// Targets:  [[10], [20], [30], ..., [num_timesteps*10]]
    /// RecordIDs: "1", "2", ..., "{num_timesteps}"
    fn create_simple_dataset(
        num_timesteps: usize,
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> SequenceDataset {
        let features: Vec<Vec<f32>> = (1..=num_timesteps as i32).map(|i| vec![i as f32]).collect();
        let targets: Vec<Vec<f32>> = (1..=num_timesteps as i32)
            .map(|i| vec![(i * 10) as f32])
            .collect();
        let record_ids: Vec<String> = (1..=num_timesteps as i32).map(|i| i.to_string()).collect();

        SequenceDataset::from_features_and_targets(
            features,
            targets,
            record_ids,
            sequence_length,
            prediction_horizon,
        )
    }

    #[test]
    fn test_dataset_builder_separates_features_and_targets() {
        // Create simple pipelines (no preprocessing for clarity)
        let mut feature_pipelines = HashMap::new();
        feature_pipelines.insert("feat1".to_string(), Pipeline::new(vec![]));
        feature_pipelines.insert("feat2".to_string(), Pipeline::new(vec![]));

        let mut target_pipelines = HashMap::new();
        target_pipelines.insert("target".to_string(), Pipeline::new(vec![]));

        let mut builder = DatasetBuilder::new(
            feature_pipelines,
            vec!["feat1".to_string(), "feat2".to_string()],
            vec!["col1".to_string(), "col2".to_string()],
            target_pipelines,
            vec!["target".to_string()],
            vec!["col3".to_string()],
            None,
        );

        // Push simple records: [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...
        for i in 0..5 {
            let mut record = HashMap::new();
            record.insert("col1".to_string(), (i * 3 + 1) as f32);
            record.insert("col2".to_string(), (i * 3 + 2) as f32);
            record.insert("col3".to_string(), (i * 3 + 3) as f32);
            builder.push(record, i.to_string()).unwrap();
        }

        // Verify features and targets are separate
        assert_eq!(builder.features.len(), 5);
        assert_eq!(builder.targets.len(), 5);

        // Check first timestep: features=[1.0, 2.0], target=[3.0]
        assert_eq!(builder.features[0], vec![1.0, 2.0]);
        assert_eq!(builder.targets[0], vec![3.0]);

        // Check second timestep: features=[4.0, 5.0], target=[6.0]
        assert_eq!(builder.features[1], vec![4.0, 5.0]);
        assert_eq!(builder.targets[1], vec![6.0]);
    }

    #[test]
    fn test_sequence_dataset_uses_separate_targets() {
        // Features: [[1, 2], [3, 4], [5, 6], [7, 8]]
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        // Targets: [[10], [20], [30], [40]]
        let targets = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0]];

        let record_ids = vec![
            "0".to_string(),
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
        ];
        let dataset = SequenceDataset::from_features_and_targets(
            features, targets, record_ids, 2, // sequence_length
            0, // prediction_horizon (predict immediately after sequence)
        );

        // Should have 3 items:
        // Item 0: seq[0,1] -> target_index = end_index(1) + horizon(0) = 1 -> targets[1] = [20]
        // Item 1: seq[1,2] -> target_index = end_index(2) + horizon(0) = 2 -> targets[2] = [30]
        // Item 2: seq[2,3] -> target_index = end_index(3) + horizon(0) = 3 -> targets[3] = [40]
        assert_eq!(dataset.len(), 3);

        // First item: sequence=[[1,2], [3,4]], target=[20]
        let item0 = dataset.get(0).unwrap();
        assert_eq!(item0.sequence, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(item0.target, vec![20.0]);

        // Second item: sequence=[[3,4], [5,6]], target=[30]
        let item1 = dataset.get(1).unwrap();
        assert_eq!(item1.sequence, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(item1.target, vec![30.0]);

        // Third item: sequence=[[5,6], [7,8]], target=[40]
        let item2 = dataset.get(2).unwrap();
        assert_eq!(item2.sequence, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        assert_eq!(item2.target, vec![40.0]);
    }

    #[test]
    fn test_get_and_metadata_h1_item0() {
        // sequence_length = 3, horizon = 1
        let ds = create_simple_dataset(12, 3, 1);
        // First item: sequence at indices [0,1,2] => [[1],[2],[3]], anchor at index 2
        // target_index = end_index + horizon = 2 + 1 = 3 => targets[3] = [40]
        let item = ds.get(0).expect("item 0 exists");
        let md = ds.get_metadata(0).expect("metadata 0 exists");

        assert_eq!(item.sequence, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(item.target, vec![40.0]); // 4th target value (index 3) = 40

        // Metadata row numbers (1-based to match data indices)
        assert_eq!(md.sequence_start_row_id, "1");
        assert_eq!(md.sequence_end_row_id, "3");
        assert_eq!(md.target_row_id, "4");

        // Anchor target should be target[end_index] = targets[2] = [30]
        assert_eq!(md.target_at_sequence_end, vec![30.0]);
    }

    #[test]
    fn test_get_and_metadata_h1_item1() {
        let ds = create_simple_dataset(12, 3, 1);
        // Second item: sequence at indices [1,2,3] => [[2],[3],[4]], anchor at index 3
        // target_index = 3 + 1 = 4 => targets[4] = [50]
        let item = ds.get(1).expect("item 1 exists");
        let md = ds.get_metadata(1).expect("metadata 1 exists");

        assert_eq!(item.sequence, vec![vec![2.0], vec![3.0], vec![4.0]]);
        assert_eq!(item.target, vec![50.0]);

        assert_eq!(md.sequence_start_row_id, "2");
        assert_eq!(md.sequence_end_row_id, "4");
        assert_eq!(md.target_row_id, "5");

        assert_eq!(md.target_at_sequence_end, vec![40.0]);
    }

    #[test]
    fn test_get_and_metadata_h3_item0() {
        // sequence_length = 3, horizon = 3
        let ds = create_simple_dataset(12, 3, 3);
        // First item: sequence at indices [0,1,2] => [[1],[2],[3]], anchor at index 2
        // target_index = end_index + horizon = 2 + 3 = 5 => targets[5] = [60]
        let item = ds.get(0).expect("item 0 exists");
        let md = ds.get_metadata(0).expect("metadata 0 exists");

        assert_eq!(item.sequence, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(item.target, vec![60.0]);

        assert_eq!(md.sequence_start_row_id, "1");
        assert_eq!(md.sequence_end_row_id, "3");
        assert_eq!(md.target_row_id, "6");

        assert_eq!(md.target_at_sequence_end, vec![30.0]);
    }
}
