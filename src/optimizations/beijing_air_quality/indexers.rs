use super::ingest;
use super::phenotype::BeijingPhenotype;
use crate::core::ingestion::Extract;
use fx_durable_ga::models::{EncodeInput, TypeName};
use fx_durable_ga::services::indexing::TrainModelConfig;
use fx_durable_ga::services::indexing::encoder::dataset::{
    SequenceDataSource, SequenceDataset, SequenceSample,
};
use fx_durable_ga::services::indexing::encoder::train::AutoencoderTrainConfig;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Handle;
use tokio::task::block_in_place;

pub struct BeijingIndexer;

const TRAIN_TIME_STEPS: usize = 120;
const TRAIN_WINDOW_STRIDE: usize = 6;
const TRAIN_MAX_WINDOWS: usize = 500;
const TRAIN_NUM_PIPELINES: usize = 32;
const TRAIN_RNG_SEED: u64 = 42;

const INFER_TIME_STEPS: usize = 120;
const INFER_WINDOW_STRIDE: usize = 12;
const INFER_MAX_WINDOWS: usize = 100;

const LATENT_SIZE: usize = 64;
const HIDDEN_SIZE: usize = 32;
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 20;
const LEARNING_RATE: f64 = 1e-3;

impl TypeName for BeijingIndexer {
    fn type_name(&self) -> &str {
        BeijingPhenotype::TYPE_NAME
    }
}

impl fx_durable_ga::models::indexable::Indexer for BeijingIndexer {
    type Type = BeijingPhenotype;

    fn dataset(
        &self,
    ) -> Arc<dyn fx_durable_ga::services::indexing::encoder::dataset::SequenceDataSource> {
        let mut rng = StdRng::seed_from_u64(TRAIN_RNG_SEED);
        let mut samples = Vec::new();

        for _ in 0..TRAIN_NUM_PIPELINES {
            let phenotype = BeijingPhenotype::random(&mut rng);
            let windows = build_transposed_windows(
                &phenotype.features,
                TRAIN_TIME_STEPS,
                TRAIN_WINDOW_STRIDE,
                TRAIN_MAX_WINDOWS,
            );

            for window in windows {
                samples.push(SequenceSample { steps: window });
            }
        }

        Arc::new(BeijingIndexingDataset::new(samples, TRAIN_TIME_STEPS))
    }

    fn training_config(&self) -> &fx_durable_ga::services::indexing::TrainModelConfig {
        static TRAIN_CONFIG: OnceLock<TrainModelConfig> = OnceLock::new();
        TRAIN_CONFIG.get_or_init(|| {
            TrainModelConfig::Lstm(AutoencoderTrainConfig {
                input_size: TRAIN_TIME_STEPS,
                hidden_size: HIDDEN_SIZE,
                latent_size: LATENT_SIZE,
                batch_size: BATCH_SIZE,
                epochs: EPOCHS,
                learning_rate: LEARNING_RATE,
            })
        })
    }

    fn preprocess(&self, entity: &Self::Type) -> fx_durable_ga::models::EncodeInput {
        let windows = build_transposed_windows(
            &entity.features,
            INFER_TIME_STEPS,
            INFER_WINDOW_STRIDE,
            INFER_MAX_WINDOWS,
        );

        if windows.is_empty() {
            return EncodeInput {
                values: Vec::new(),
                dimensions: vec![0, INFER_TIME_STEPS],
            };
        }

        let mut stacked = Vec::new();
        for window in windows {
            stacked.extend(window);
        }

        let seq_len = stacked.len();
        let input_size = stacked.first().map(|row| row.len()).unwrap_or(0);

        EncodeInput {
            values: flatten(&stacked),
            dimensions: vec![seq_len, input_size],
        }
    }
}

#[derive(Clone)]
struct BeijingIndexingDataset {
    dataset: SequenceDataset,
}

impl BeijingIndexingDataset {
    fn new(samples: Vec<SequenceSample>, time_steps: usize) -> Self {
        Self {
            dataset: SequenceDataset::new(samples, time_steps),
        }
    }
}

impl SequenceDataSource for BeijingIndexingDataset {
    fn checksum(&self) -> Vec<u8> {
        self.dataset.checksum()
    }

    fn sample(&self, index: usize) -> Option<SequenceSample> {
        self.dataset.sample(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

fn build_transposed_windows(
    features: &[Extract],
    time_steps: usize,
    window_stride: usize,
    max_windows: usize,
) -> Vec<Vec<Vec<f32>>> {
    let sequences = block_on_ingest(features);
    let total = sequences.len(time_steps, 0);
    let stride = window_stride.max(1);
    let mut windows = Vec::new();

    for index in (0..total).step_by(stride) {
        if windows.len() >= max_windows {
            break;
        }

        let Some((window, _)) = sequences.get_item(index, time_steps, 0) else {
            continue;
        };

        if window.is_empty() {
            continue;
        }

        let transposed = transpose(window);
        if transposed.is_empty() {
            continue;
        }

        windows.push(transposed);
    }

    windows
}

fn block_on_ingest(features: &[Extract]) -> crate::core::ingestion::ManySequences {
    let targets: Vec<Extract> = Vec::new();
    match Handle::try_current() {
        Ok(handle) => block_in_place(|| {
            handle
                .block_on(ingest(features, &targets))
                .expect("Failed to ingest Beijing dataset for indexing")
        }),
        Err(_) => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build tokio runtime for ingestion");

            runtime
                .block_on(ingest(features, &targets))
                .expect("Failed to ingest Beijing dataset for indexing")
        }
    }
}

fn flatten(matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    if cols == 0 {
        return Vec::new();
    }

    let mut transposed = vec![vec![0.0f32; rows]; cols];

    for (r, row) in matrix.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            transposed[c][r] = *value;
        }
    }

    transposed
}
