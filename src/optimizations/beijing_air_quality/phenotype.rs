use fx_durable_ga::models::{Encodeable, GeneBounds};
use serde::{Deserialize, Serialize};

use super::ingestion::SOURCE_COLUMNS;
use crate::preprocessor::{Cos, Node, Pipeline, Roc, Sin, Std, ZScore};

// Constants for gene bounds
const FEATURE_COUNT: usize = 7;
const SOURCE_COLUMN_COUNT: usize = 15; // Length of SOURCE_COLUMNS
const MAX_PIPELINE_LENGTH: usize = 2; // Max transforms per feature
const TRANSFORM_COUNT: usize = 12; // Number of Transform enum variants

const HIDDEN_SIZE_OPTIONS: &[usize] = &[4, 8, 16, 32, 64, 128];
const LEARNING_RATE_OPTIONS: &[f64] = &[1e-4, 5e-4, 1e-3];
const SEQUENCE_LENGTH_OPTIONS: &[usize] = &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

/// Transform variants that map genes to specific preprocessing operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Transform {
    ZScore10,
    ZScore24,
    ZScore48,
    ZScore96,
    Roc1,
    Roc4,
    Roc8,
    Roc12,
    Std10,
    Std24,
    Std48,
    Std96,
}

impl Transform {
    fn from_gene(gene: i64) -> Option<Self> {
        match gene {
            0 => Some(Self::ZScore10),
            1 => Some(Self::ZScore24),
            2 => Some(Self::ZScore48),
            3 => Some(Self::ZScore96),
            4 => Some(Self::Roc1),
            5 => Some(Self::Roc4),
            6 => Some(Self::Roc8),
            7 => Some(Self::Roc12),
            8 => Some(Self::Std10),
            9 => Some(Self::Std24),
            10 => Some(Self::Std48),
            11 => Some(Self::Std96),
            _ => None,
        }
    }

    fn to_gene(self) -> i64 {
        match self {
            Self::ZScore10 => 0,
            Self::ZScore24 => 1,
            Self::ZScore48 => 2,
            Self::ZScore96 => 3,
            Self::Roc1 => 4,
            Self::Roc4 => 5,
            Self::Roc8 => 6,
            Self::Roc12 => 7,
            Self::Std10 => 8,
            Self::Std24 => 9,
            Self::Std48 => 10,
            Self::Std96 => 11,
        }
    }

    /// Convert Transform to a preprocessor Node
    fn to_node(&self) -> Node {
        match self {
            Self::ZScore10 => Node::ZScore(ZScore::new(10)),
            Self::ZScore24 => Node::ZScore(ZScore::new(24)),
            Self::ZScore48 => Node::ZScore(ZScore::new(48)),
            Self::ZScore96 => Node::ZScore(ZScore::new(96)),
            Self::Roc1 => Node::Roc(Roc::new(1)),
            Self::Roc4 => Node::Roc(Roc::new(4)),
            Self::Roc8 => Node::Roc(Roc::new(8)),
            Self::Roc12 => Node::Roc(Roc::new(12)),
            Self::Std10 => Node::Std(Std::new(10)),
            Self::Std24 => Node::Std(Std::new(24)),
            Self::Std48 => Node::Std(Std::new(48)),
            Self::Std96 => Node::Std(Std::new(96)),
        }
    }
}

/// A feature with its source column and preprocessing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Feature {
    pub(super) source: String,             // e.g., "TEMP", "PRES"
    pub(super) transforms: Vec<Transform>, // Pipeline of transforms to apply
}

impl Feature {
    /// Convert feature's transforms to a Pipeline
    pub(super) fn to_pipeline(&self) -> Pipeline {
        let nodes: Vec<Node> = self.transforms.iter().map(|t| t.to_node()).collect();
        Pipeline::new(nodes)
    }
}

/// Phenotype for Beijing air quality prediction optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeijingPhenotype {
    #[serde(skip)]
    features: Vec<Feature>, // The optimized features with pipelines
    pub hidden_size: usize,
    pub learning_rate: f64,
    pub sequence_length: usize,
}

impl BeijingPhenotype {
    /// Get a reference to the features (useful for evaluator)
    pub fn features(&self) -> &[Feature] {
        &self.features
    }
}

impl Encodeable for BeijingPhenotype {
    const NAME: &'static str = "beijing_air_quality_feature_engineering";

    type Phenotype = BeijingPhenotype;

    fn morphology() -> Vec<GeneBounds> {
        let mut bounds = Vec::new();

        // 7 features, each with 4 genes
        for _ in 0..FEATURE_COUNT {
            // Source column index (0 to SOURCE_COLUMN_COUNT-1)
            bounds.push(
                GeneBounds::integer(
                    0,
                    (SOURCE_COLUMN_COUNT - 1) as i32,
                    SOURCE_COLUMN_COUNT as u32,
                )
                .unwrap(),
            );

            // Pipeline length (0, 1, or 2 transforms)
            bounds.push(
                GeneBounds::integer(
                    0,
                    MAX_PIPELINE_LENGTH as i32,
                    (MAX_PIPELINE_LENGTH + 1) as u32,
                )
                .unwrap(),
            );

            // Transform 1 index (0 to TRANSFORM_COUNT-1)
            bounds.push(
                GeneBounds::integer(0, (TRANSFORM_COUNT - 1) as i32, TRANSFORM_COUNT as u32)
                    .unwrap(),
            );

            // Transform 2 index (0 to TRANSFORM_COUNT-1)
            bounds.push(
                GeneBounds::integer(0, (TRANSFORM_COUNT - 1) as i32, TRANSFORM_COUNT as u32)
                    .unwrap(),
            );
        }

        // Hyperparameters - each stores an index into the options array
        bounds.push(
            GeneBounds::integer(
                0,
                (HIDDEN_SIZE_OPTIONS.len() - 1) as i32,
                HIDDEN_SIZE_OPTIONS.len() as u32,
            )
            .unwrap(),
        );
        bounds.push(
            GeneBounds::integer(
                0,
                (LEARNING_RATE_OPTIONS.len() - 1) as i32,
                LEARNING_RATE_OPTIONS.len() as u32,
            )
            .unwrap(),
        );
        bounds.push(
            GeneBounds::integer(
                0,
                (SEQUENCE_LENGTH_OPTIONS.len() - 1) as i32,
                SEQUENCE_LENGTH_OPTIONS.len() as u32,
            )
            .unwrap(),
        );

        bounds
    }

    fn encode(&self) -> Vec<i64> {
        let mut genes = Vec::new();

        // Encode 7 features
        for feature in &self.features {
            // Source column
            let source_idx = SOURCE_COLUMNS
                .iter()
                .position(|&col| col == feature.source)
                .unwrap_or(0) as i64;
            genes.push(source_idx);

            // Pipeline length (max 2)
            genes.push(feature.transforms.len().min(MAX_PIPELINE_LENGTH) as i64);

            // Transforms (pad with 0 if fewer than 2)
            for i in 0..MAX_PIPELINE_LENGTH {
                if i < feature.transforms.len() {
                    genes.push(feature.transforms[i].to_gene());
                } else {
                    genes.push(0);
                }
            }
        }

        // Encode hyperparameters using position in options arrays
        let hidden_size_idx = HIDDEN_SIZE_OPTIONS
            .iter()
            .position(|&size| size == self.hidden_size)
            .unwrap_or(2) as i64; // default to index 2 (16)
        genes.push(hidden_size_idx);

        let lr_idx = LEARNING_RATE_OPTIONS
            .iter()
            .position(|&lr| (lr - self.learning_rate).abs() < 1e-10)
            .unwrap_or(1) as i64; // default to index 1 (5e-4)
        genes.push(lr_idx);

        let seq_len_idx = SEQUENCE_LENGTH_OPTIONS
            .iter()
            .position(|&len| len == self.sequence_length)
            .unwrap_or(4) as i64; // default to index 4 (50)
        genes.push(seq_len_idx);

        genes
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        let mut features = Vec::new();

        // Decode 7 features (4 genes per feature: source, length, t1, t2)
        for i in 0..FEATURE_COUNT {
            let base_idx = i * 4;

            let source_idx = genes[base_idx].clamp(0, SOURCE_COLUMN_COUNT as i64 - 1) as usize;
            let source = SOURCE_COLUMNS[source_idx].to_string();

            let pipeline_length = genes[base_idx + 1].clamp(0, MAX_PIPELINE_LENGTH as i64) as usize;

            let mut transforms = Vec::new();
            for j in 0..pipeline_length {
                if let Some(transform) = Transform::from_gene(genes[base_idx + 2 + j]) {
                    transforms.push(transform);
                }
            }

            features.push(Feature { source, transforms });
        }

        // Decode hyperparameters (genes are at indices 28, 29, 30)
        let hidden_size_idx = genes[28].clamp(0, HIDDEN_SIZE_OPTIONS.len() as i64 - 1) as usize;
        let hidden_size = HIDDEN_SIZE_OPTIONS[hidden_size_idx];

        let lr_idx = genes[29].clamp(0, LEARNING_RATE_OPTIONS.len() as i64 - 1) as usize;
        let learning_rate = LEARNING_RATE_OPTIONS[lr_idx];

        let seq_len_idx = genes[30].clamp(0, SEQUENCE_LENGTH_OPTIONS.len() as i64 - 1) as usize;
        let sequence_length = SEQUENCE_LENGTH_OPTIONS[seq_len_idx];

        BeijingPhenotype {
            features,
            hidden_size,
            learning_rate,
            sequence_length,
        }
    }
}
