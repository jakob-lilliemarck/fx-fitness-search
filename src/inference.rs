use crate::batcher::SequenceBatcher;
use crate::dataset::{SequenceDatasetItem, Timestep};
use crate::model::{FeedForward, SequenceModel};
use crate::train_config::TrainConfig;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};

pub struct InferenceEngine<B: Backend> {
    pub model: FeedForward<B>,
    pub config: TrainConfig,
    device: B::Device,
}

impl<B: Backend> InferenceEngine<B> {
    /// Load model and config from disk
    pub fn load(model_path: &str, device: &B::Device) -> anyhow::Result<Self> {
        // Load config from {model_path}.config.json
        let config_path = format!("{}.config.json", model_path);
        let config = TrainConfig::load(&config_path)?;

        // Load model record from {model_path}
        let record = CompactRecorder::new()
            .load(model_path.into(), device)
            .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?;

        // Instantiate FeedForward with config params
        let model = FeedForward::<B>::new(
            device,
            config.input_size,
            config.hidden_size,
            config.output_size,
            config.sequence_length,
        );

        // Load record into model
        let model = model.load_record(record);

        Ok(Self {
            model,
            config,
            device: device.clone(),
        })
    }

    /// Make a single prediction given a sequence of timesteps
    pub fn predict(&self, sequence: Vec<Timestep>) -> anyhow::Result<Vec<f32>> {
        // Create SequenceDatasetItem from sequence
        let item = SequenceDatasetItem {
            sequence,
            target: vec![0.0; self.config.output_size], // Placeholder, not used in inference
        };

        // Batch it (batch size = 1)
        // The batcher is a zero-sized struct, instantiating it here is essentially
        // free and lets us keep the code consistent witht the training code.
        let batcher = SequenceBatcher::<B>::new();
        let batch = batcher.batch(vec![item], &self.device);

        // Call model.forward()
        let output = self.model.forward(batch.sequences);

        // Extract prediction (convert to Vec<f32>)
        let output_data = output.into_data();
        let prediction = output_data
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert tensor to vec: {:?}", e))?;

        Ok(prediction)
    }
}
