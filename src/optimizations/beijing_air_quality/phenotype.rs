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
    type Phenotype = Node;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 100;
        let steps = 101;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            Node::Noop => 0,
            Node::Roc(Roc { offset, .. }) => {
                if *offset > 100 {
                    return Err(Error::Encode(format!("Unexpected Roc offset: {}", offset)));
                }
                *offset as i64
            }
            _ => return Err(Error::Encode(format!("Unexpected node type"))),
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let offset = genes[0] as usize;
        if offset == 0 {
            Ok(Node::Noop)
        } else {
            Ok(Node::Roc(Roc::new(offset)))
        }
    }
}

impl Genotype for Std {
    type Phenotype = Node;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 100;
        let steps = 101;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            Node::Noop => 0,
            Node::Std(Std { window, .. }) => {
                if *window > 100 {
                    return Err(Error::Encode(format!("Unexpected Std window: {}", window)));
                }
                *window as i64
            }
            _ => return Err(Error::Encode(format!("Unexpected node type"))),
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let window = genes[0] as usize;
        if window == 0 {
            Ok(Node::Noop)
        } else {
            Ok(Node::Std(Std::new(window)))
        }
    }
}

impl Genotype for Ema {
    type Phenotype = Node;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 100;
        let steps = 101;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            Node::Noop => 0,
            Node::Ema(Ema { window, .. }) => {
                if *window > 100 {
                    return Err(Error::Encode(format!("Unexpected Ema window: {}", window)));
                }
                *window as i64
            }
            _ => return Err(Error::Encode(format!("Unexpected node type"))),
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let window = genes[0] as usize;
        if window == 0 {
            Ok(Node::Noop)
        } else {
            let alpha = 2.0 / (window + 1) as f32;
            Ok(Node::Ema(Ema::new(window, alpha)))
        }
    }
}

impl Genotype for ZScore {
    type Phenotype = Node;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let lower = 0;
        let upper = 100;
        let steps = 101;
        let bounds = GeneBounds::integer(lower, upper, steps)?;
        Ok(vec![bounds])
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let gene = match phenotype {
            Node::Noop => 0,
            Node::ZScore(ZScore { window, .. }) => {
                if *window > 100 {
                    return Err(Error::Encode(format!(
                        "Unexpected ZScore window: {}",
                        window
                    )));
                }
                *window as i64
            }
            _ => return Err(Error::Encode(format!("Unexpected node type"))),
        };
        Ok(vec![gene])
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 1 {
            return Err(Error::Decode(
                "Could not decode genes of unexpected length".to_string(),
            ));
        }

        let window = genes[0] as usize;
        if window == 0 {
            Ok(Node::Noop)
        } else {
            Ok(Node::ZScore(ZScore::new(window)))
        }
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

impl Genotype for Extract {
    type Phenotype = Extract;

    fn morphology() -> Result<Vec<GeneBounds>, Error> {
        let mut bounds = Vec::with_capacity(10);
        bounds.extend(Optimizable::morphology()?);
        bounds.extend(Interpolation::morphology()?);
        // Layer 1: ZScore, Roc, Std, Ema
        bounds.extend(ZScore::morphology()?);
        bounds.extend(Roc::morphology()?);
        bounds.extend(Std::morphology()?);
        bounds.extend(Ema::morphology()?);
        // Layer 2: ZScore, Roc, Std, Ema
        bounds.extend(ZScore::morphology()?);
        bounds.extend(Roc::morphology()?);
        bounds.extend(Std::morphology()?);
        bounds.extend(Ema::morphology()?);
        Ok(bounds)
    }

    fn encode(phenotype: &Self::Phenotype) -> Result<Vec<i64>, Error> {
        let mut genes = Vec::with_capacity(10);

        let optimizable: Optimizable = phenotype
            .source
            .as_str()
            .try_into()
            .map_err(|err: OptimizableError| Error::Encode(err.to_string()))?;

        genes.extend(Optimizable::encode(&optimizable)?);
        genes.extend(Interpolation::encode(&phenotype.interpolation)?);

        // Encode 8 nodes: [zscore_l1, roc_l1, std_l1, ema_l1, zscore_l2, roc_l2, std_l2, ema_l2]
        if phenotype.nodes.len() != 8 {
            return Err(Error::Encode(format!(
                "Expected 8 nodes, got {}",
                phenotype.nodes.len()
            )));
        }

        // Layer 1
        genes.extend(ZScore::encode(&phenotype.nodes[0])?);
        genes.extend(Roc::encode(&phenotype.nodes[1])?);
        genes.extend(Std::encode(&phenotype.nodes[2])?);
        genes.extend(Ema::encode(&phenotype.nodes[3])?);
        // Layer 2
        genes.extend(ZScore::encode(&phenotype.nodes[4])?);
        genes.extend(Roc::encode(&phenotype.nodes[5])?);
        genes.extend(Std::encode(&phenotype.nodes[6])?);
        genes.extend(Ema::encode(&phenotype.nodes[7])?);

        Ok(genes)
    }

    fn decode(genes: &[i64]) -> Result<Self::Phenotype, Error> {
        if genes.len() != 10 {
            return Err(Error::Decode(format!(
                "Expected 10 genes, got {}",
                genes.len()
            )));
        }

        let source = Optimizable::decode(&[genes[0]])?.to_string();
        let interpolation = Interpolation::decode(&[genes[1]])?;

        // Decode 8 nodes: [zscore_l1, roc_l1, std_l1, ema_l1, zscore_l2, roc_l2, std_l2, ema_l2]
        let nodes = vec![
            // Layer 1
            ZScore::decode(&[genes[2]])?,
            Roc::decode(&[genes[3]])?,
            Std::decode(&[genes[4]])?,
            Ema::decode(&[genes[5]])?,
            // Layer 2
            ZScore::decode(&[genes[6]])?,
            Roc::decode(&[genes[7]])?,
            Std::decode(&[genes[8]])?,
            Ema::decode(&[genes[9]])?,
        ];

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

        // 7 features, each with Extract morphology
        for _ in 0..FEATURE_COUNT {
            bounds.extend(Extract::morphology().expect("Failed to get Extract morphology"));
        }

        // Hyperparameters
        bounds.extend(HiddenSize::morphology().expect("Failed to get HiddenSize morphology"));
        bounds.extend(LearningRate::morphology().expect("Failed to get LearningRate morphology"));
        bounds
            .extend(SequenceLength::morphology().expect("Failed to get SequenceLength morphology"));

        bounds
    }

    fn encode(&self) -> Vec<i64> {
        let mut genes = Vec::new();

        // Encode 7 features
        for extract in &self.features {
            genes.extend(Extract::encode(extract).expect("Failed to encode Extract"));
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

        // Decode 7 features
        for _ in 0..FEATURE_COUNT {
            let extract_morphology =
                Extract::morphology().expect("Failed to get Extract morphology");
            let extract_gene_count = extract_morphology.len();
            assert!(
                gene_offset + extract_gene_count <= genes.len(),
                "Not enough genes to decode Extract"
            );
            let extract = Extract::decode(&genes[gene_offset..gene_offset + extract_gene_count])
                .expect("Failed to decode Extract");
            features.push(extract);
            gene_offset += extract_gene_count;
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
