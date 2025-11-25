use crate::core::ingestion::Extract;
use crate::core::interpolation::{Interpolation, LinearInterpolator};
use crate::core::preprocessor::{Ema, Node, Roc, Std, ZScore};
use crate::optimizations::beijing_air_quality::cast::{Optimizable, OptimizableError};
use fx_durable_ga::models::{Encodeable, GeneBoundError, GeneBounds};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("DecodeError: {0}")]
    Decode(String),
    #[error("EncodeError: {0}")]
    Encode(String),
    #[error("MorphologyError: {0}")]
    Morphology(#[from] GeneBoundError),
}

pub trait Genotype {
    type Phenotype;
    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error>;
    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error>;
    fn morphology() -> Result<Vec<GeneBounds>, Error>;
}

impl Genotype for Roc {
    type Phenotype = Roc;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 5;
        let steps = 6;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype.offset {
            2 => 0,
            4 => 1,
            8 => 2,
            16 => 3,
            32 => 4,
            64 => 5,
            _ => {
                return Err(Error::Encode(format!(
                    "Unexpected Roc offset: {}",
                    phenotype.offset
                )));
            }
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let offset = match genes[0] {
            0 => 2,
            1 => 4,
            2 => 8,
            3 => 16,
            4 => 32,
            5 => 64,
            _ => return Err(Error::Decode(format!("Unexpected Roc gene: {}", genes[0]))),
        };
        Ok(Roc::new(offset))
    }
}

impl Genotype for Std {
    type Phenotype = Std;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 5;
        let steps = 6;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype.window {
            2 => 0,
            4 => 1,
            8 => 2,
            16 => 3,
            32 => 4,
            64 => 5,
            _ => {
                return Err(Error::Encode(format!(
                    "Unexpected Std window: {}",
                    phenotype.window
                )));
            }
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let window = match genes[0] {
            0 => 2,
            1 => 4,
            2 => 8,
            3 => 16,
            4 => 32,
            5 => 64,
            _ => return Err(Error::Decode(format!("Unexpected Std gene: {}", genes[0]))),
        };
        Ok(Std::new(window))
    }
}

impl Genotype for Ema {
    type Phenotype = Ema;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 5;
        let steps = 6;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype.window {
            2 => 0,
            4 => 1,
            8 => 2,
            16 => 3,
            32 => 4,
            64 => 5,
            _ => {
                return Err(Error::Encode(format!(
                    "Unexpected Ema window: {}",
                    phenotype.window
                )));
            }
        };

        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let window = match genes[0] {
            0 => 2,
            1 => 4,
            2 => 8,
            3 => 16,
            4 => 32,
            5 => 64,
            _ => return Err(Error::Decode(format!("Unexpected Ema gene: {}", genes[0]))),
        };
        let alpha = 2.0 / (window + 1) as f32;
        Ok(Ema::new(window, alpha))
    }
}

impl Genotype for ZScore {
    type Phenotype = ZScore;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 5;
        let steps = 6;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype.window {
            2 => 0,
            4 => 1,
            8 => 2,
            16 => 3,
            32 => 4,
            64 => 5,
            _ => {
                return Err(Error::Encode(format!(
                    "Unexpected ZScore window: {}",
                    phenotype.window
                )));
            }
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let window = match genes[0] {
            0 => 2,
            1 => 4,
            2 => 8,
            3 => 16,
            4 => 32,
            5 => 64,
            _ => {
                return Err(Error::Decode(format!(
                    "Unexpected ZScore gene: {}",
                    genes[0]
                )));
            }
        };
        Ok(ZScore::new(window))
    }
}

impl Genotype for Interpolation {
    type Phenotype = Interpolation;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 4;
        let steps = 5;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            Interpolation::Linear(linear) => match linear.window_size {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                16 => 4,
                _ => {
                    return Err(Error::Encode(
                        "Could not encode unexpected linear interpolation window to gene"
                            .to_string(),
                    ));
                }
            },
            _ => {
                return Err(Error::Encode(
                    "Could not encode unexpected interpolation variant to gene".to_string(),
                ));
            }
        };

        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        let interpolation = match genes[0] {
            0 => Interpolation::Linear(LinearInterpolator::new(1)),
            1 => Interpolation::Linear(LinearInterpolator::new(2)),
            2 => Interpolation::Linear(LinearInterpolator::new(4)),
            3 => Interpolation::Linear(LinearInterpolator::new(8)),
            4 => Interpolation::Linear(LinearInterpolator::new(16)),
            _ => {
                return Err(Error::Decode(
                    "Could not decode unexpected gene value to Interpolation".to_string(),
                ));
            }
        };

        Ok(interpolation)
    }
}

pub struct Toggle<T: Genotype> {
    _source: T,
}

impl<T: Genotype> Genotype for Toggle<T> {
    type Phenotype = (bool, T::Phenotype);

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let mut bounds = vec![GeneBounds::integer(0, 1, 2)?]; // on/off
        bounds.extend(T::morphology()?);
        Ok(bounds)
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let mut genes = vec![if phenotype.0 { 1 } else { 0 }];
        genes.extend(T::encode(&phenotype.1)?);
        Ok(genes)
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.is_empty() {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }
        let enabled = genes[0] != 0;
        let inner = T::decode(&genes[1..])?;
        Ok((enabled, inner))
    }
}

impl Genotype for Optimizable {
    type Phenotype = Optimizable;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 11;
        let steps = 12;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            Self::Pm25 => 0,
            Self::Pm10 => 1,
            Self::So2 => 2,
            Self::No2 => 3,
            Self::Co => 4,
            Self::O3 => 5,
            Self::Temp => 6,
            Self::Pres => 7,
            Self::Dewp => 8,
            Self::Rain => 9,
            Self::Wspm => 10,
            Self::Wd => 11,
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let key = match genes[0] {
            0 => Optimizable::Pm25,
            1 => Optimizable::Pm10,
            2 => Optimizable::So2,
            3 => Optimizable::No2,
            4 => Optimizable::Co,
            5 => Optimizable::O3,
            6 => Optimizable::Temp,
            7 => Optimizable::Pres,
            8 => Optimizable::Dewp,
            9 => Optimizable::Rain,
            10 => Optimizable::Wspm,
            11 => Optimizable::Wd,
            _ => {
                return Err(Error::Decode(
                    "Could not decode unexpected gene value to Optimizable".to_string(),
                ));
            }
        };

        Ok(key)
    }
}

pub enum NodeType {
    Std,
    Ema,
    Roc,
    ZScore,
}

impl Genotype for NodeType {
    type Phenotype = NodeType;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 3;
        let steps = 4;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            NodeType::Std => 0,
            NodeType::Roc => 1,
            NodeType::Ema => 2,
            NodeType::ZScore => 3,
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let node_type = match genes[0] {
            0 => NodeType::Std,
            1 => NodeType::Roc,
            2 => NodeType::Ema,
            3 => NodeType::ZScore,
            _ => {
                return Err(Error::Decode(
                    "Could not decode unexpected gene value to Optimizable".to_string(),
                ));
            }
        };

        Ok(node_type)
    }
}

impl Genotype for Node {
    type Phenotype = Node;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let mut bounds = Vec::with_capacity(3);
        bounds.push(GeneBounds::integer(0, 1, 2)?); // enabled: 0-1
        bounds.extend(NodeType::morphology()?); // type: 0-3
        bounds.push(GeneBounds::integer(0, 5, 6)?); // variant: 0-5
        Ok(bounds)
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let mut genes = vec![1]; // enabled=1

        match phenotype {
            Node::ZScore(zs) => {
                genes.extend(NodeType::encode(&NodeType::ZScore)?);
                genes.extend(ZScore::encode(zs)?);
            }
            Node::Roc(roc) => {
                genes.extend(NodeType::encode(&NodeType::Roc)?);
                genes.extend(Roc::encode(roc)?);
            }
            Node::Std(std) => {
                genes.extend(NodeType::encode(&NodeType::Std)?);
                genes.extend(Std::encode(std)?);
            }
            Node::Ema(ema) => {
                genes.extend(NodeType::encode(&NodeType::Ema)?);
                genes.extend(Ema::encode(ema)?);
            }
            _ => return Err(Error::Encode("Unexpected node type".to_string())),
        }
        Ok(genes)
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 3 {
            return Err(Error::Decode(format!(
                "Expected 3 genes for Node, got {}",
                genes.len()
            )));
        }

        // If disabled (genes[0] == 0), return error—caller won't add to Vec
        if genes[0] == 0 {
            return Err(Error::Decode("Node is disabled".to_string()));
        }

        let node_type = NodeType::decode(&[genes[1]])?;

        match node_type {
            NodeType::ZScore => Ok(Node::ZScore(ZScore::decode(&[genes[2]])?)),
            NodeType::Roc => Ok(Node::Roc(Roc::decode(&[genes[2]])?)),
            NodeType::Std => Ok(Node::Std(Std::decode(&[genes[2]])?)),
            NodeType::Ema => Ok(Node::Ema(Ema::decode(&[genes[2]])?)),
        }
    }
}

impl Genotype for Extract {
    type Phenotype = Extract;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let mut bounds = Vec::new();
        bounds.extend(Optimizable::morphology()?); // source: 1 gene
        bounds.extend(Interpolation::morphology()?); // interpolation: 1 gene
        // Layer 1: 3 genes (enabled, type, variant)
        bounds.extend(Node::morphology()?);
        // Layer 2: 3 genes (enabled, type, variant)
        bounds.extend(Node::morphology()?);
        Ok(bounds)
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let mut genes = Vec::new();

        let optimizable: Optimizable = phenotype
            .source
            .as_str()
            .try_into()
            .map_err(|err: OptimizableError| Error::Encode(err.to_string()))?;

        genes.extend(Optimizable::encode(&optimizable)?);
        genes.extend(Interpolation::encode(&phenotype.interpolation)?);

        // Encode up to 2 enabled layers
        let mut layer_idx = 0;
        for node in &phenotype.nodes {
            if layer_idx >= 2 {
                break;
            }
            genes.extend(Node::encode(node)?);
            layer_idx += 1;
        }

        // Pad remaining layers with disabled (0, 0, 0)
        for _ in layer_idx..2 {
            genes.push(0); // enabled=0
            genes.push(0); // type (ignored)
            genes.push(0); // variant (ignored)
        }

        Ok(genes)
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 8 {
            // 1 (source) + 1 (interpolation) + 3 (layer 1) + 3 (layer 2)
            return Err(Error::Decode(format!(
                "Expected 8 genes, got {}",
                genes.len()
            )));
        }

        let source = Optimizable::decode(&[genes[0]])?.to_string();
        let interpolation = Interpolation::decode(&[genes[1]])?;

        let mut nodes = Vec::new();

        // Decode Layer 1 (genes 2-4)
        if genes[2] != 0 {
            // if enabled
            if let Ok(node) = Node::decode(&genes[2..5]) {
                nodes.push(node);
            }
        }

        // Decode Layer 2 (genes 5-7)
        if genes[5] != 0 {
            // if enabled
            if let Ok(node) = Node::decode(&genes[5..8]) {
                nodes.push(node);
            }
        }

        Ok(Extract {
            source,
            interpolation,
            nodes,
        })
    }
}

pub struct HiddenSize(usize);

impl Genotype for HiddenSize {
    type Phenotype = HiddenSize;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let bounds = GeneBounds::integer(0, 5, 6)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype.0 {
            4 => 0,
            8 => 1,
            16 => 2,
            32 => 3,
            64 => 4,
            128 => 5,
            _ => {
                return Err(Error::Encode(format!(
                    "Unknown hidden size: {}",
                    phenotype.0
                )));
            }
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let value = match genes[0] {
            0 => 4,
            1 => 8,
            2 => 16,
            3 => 32,
            4 => 64,
            5 => 128,
            _ => {
                return Err(Error::Decode(format!(
                    "Unknown hidden size gene: {}",
                    genes[0]
                )));
            }
        };
        Ok(HiddenSize(value))
    }
}

pub struct LearningRate(f64);

impl Genotype for LearningRate {
    type Phenotype = LearningRate;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let bounds = GeneBounds::integer(0, 3, 4)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype.0 {
            lr if (lr - 0.0001).abs() < 1e-10 => 0,
            lr if (lr - 0.0004).abs() < 1e-10 => 1,
            lr if (lr - 0.0007).abs() < 1e-10 => 2,
            lr if (lr - 0.001).abs() < 1e-10 => 3,
            _ => {
                return Err(Error::Encode(format!(
                    "Unknown learning rate: {}",
                    phenotype.0
                )));
            }
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let value = match genes[0] {
            0 => 0.0001,
            1 => 0.0004,
            2 => 0.0007,
            3 => 0.001,
            _ => {
                return Err(Error::Decode(format!(
                    "Unknown learning rate gene: {}",
                    genes[0]
                )));
            }
        };
        Ok(LearningRate(value))
    }
}

pub struct SequenceLength(usize);

impl Genotype for SequenceLength {
    type Phenotype = SequenceLength;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let bounds = GeneBounds::integer(0, 9, 10)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = (phenotype.0 / 10 - 1) as i64;
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let value = (genes[0] as usize + 1) * 10;
        Ok(SequenceLength(value))
    }
}

pub struct BeijingPhenotype {
    features: Vec<Extract>,
    pub hidden_size: usize,
    pub learning_rate: f64,
    pub sequence_length: usize,
}

impl BeijingPhenotype {
    pub fn features(&self) -> &[Extract] {
        &self.features
    }
}

// Constants for gene bounds
const FEATURE_COUNT: usize = 7;

impl Encodeable for BeijingPhenotype {
    const NAME: &'static str = "beijing_air_quality";

    type Phenotype = BeijingPhenotype;

    fn morphology() -> Vec<GeneBounds> {
        let mut bounds = Vec::new();

        // 7 features, each as Toggle<Extract>
        for _ in 0..FEATURE_COUNT {
            bounds.extend(
                Toggle::<Extract>::morphology().expect("Failed to get Toggle<Extract> morphology"),
            );
        }

        // Hyperparameters
        bounds.extend(HiddenSize::morphology().expect("Failed to get HiddenSize morphology"));
        bounds.extend(LearningRate::morphology().expect("Failed to get LearningRate morphology"));
        bounds
            .extend(SequenceLength::morphology().expect("Failed to get SequenceLength morphology"));

        bounds
    }

    fn encode(&self) -> Vec<i64> {
        use std::collections::HashSet;

        let mut genes = Vec::new();

        // Define the 7 feature sources in order
        let feature_sources = [
            Optimizable::Pm25,
            Optimizable::Pm10,
            Optimizable::So2,
            Optimizable::No2,
            Optimizable::Co,
            Optimizable::O3,
            Optimizable::Temp,
        ];

        // Build a set of enabled sources for quick lookup
        let enabled_sources: HashSet<String> = self
            .features
            .iter()
            .map(|extract| extract.source.clone())
            .collect();

        // Encode each feature slot
        for optimizable in &feature_sources {
            let source_str = optimizable.to_string();
            let enabled = enabled_sources.contains(&source_str);

            // Get the extract if enabled, otherwise create a dummy one
            let extract = self
                .features
                .iter()
                .find(|e| e.source == source_str)
                .cloned()
                .unwrap_or_else(|| Extract::new(&source_str));

            genes.extend(
                Toggle::<Extract>::encode(&(enabled, extract))
                    .expect("Failed to encode Toggle<Extract>"),
            );
        }

        // Encode hyperparameters
        genes.extend(
            HiddenSize::encode(&HiddenSize(self.hidden_size)).expect("Failed to encode HiddenSize"),
        );
        genes.extend(
            LearningRate::encode(&LearningRate(self.learning_rate))
                .expect("Failed to encode LearningRate"),
        );
        genes.extend(
            SequenceLength::encode(&SequenceLength(self.sequence_length))
                .expect("Failed to encode SequenceLength"),
        );

        genes
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        let mut features = Vec::new();
        let mut gene_offset = 0;

        // Decode 7 feature slots
        for _ in 0..FEATURE_COUNT {
            let toggle_morphology =
                Toggle::<Extract>::morphology().expect("Failed to get Toggle<Extract> morphology");
            let toggle_gene_count = toggle_morphology.len();
            assert!(
                gene_offset + toggle_gene_count <= genes.len(),
                "Not enough genes to decode Toggle<Extract>"
            );

            let (enabled, extract) =
                Toggle::<Extract>::decode(&genes[gene_offset..gene_offset + toggle_gene_count])
                    .expect("Failed to decode Toggle<Extract>");

            if enabled {
                features.push(extract);
            }

            gene_offset += toggle_gene_count;
        }

        // Decode hyperparameters
        assert!(
            gene_offset + 3 <= genes.len(),
            "Not enough genes to decode hyperparameters: need 3 more, have {}",
            genes.len() - gene_offset
        );

        let hidden_size = HiddenSize::decode(&genes[gene_offset..gene_offset + 1])
            .expect("Failed to decode HiddenSize")
            .0;
        gene_offset += 1;

        let learning_rate = LearningRate::decode(&genes[gene_offset..gene_offset + 1])
            .expect("Failed to decode LearningRate")
            .0;
        gene_offset += 1;

        let sequence_length = SequenceLength::decode(&genes[gene_offset..gene_offset + 1])
            .expect("Failed to decode SequenceLength")
            .0;

        BeijingPhenotype {
            features,
            hidden_size,
            learning_rate,
            sequence_length,
        }
    }
}
