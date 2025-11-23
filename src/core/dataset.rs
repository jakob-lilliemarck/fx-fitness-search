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
