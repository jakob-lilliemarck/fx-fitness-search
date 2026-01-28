use super::protocol::{Request, RequestVariables, Response};
use crate::core::ingestion::Extract;
use crate::core::interpolation::{Interpolation, LinearInterpolator};
use crate::core::preprocessing::{Cos, Ema, Node, Roc, Sin, Std, ZScore};
use crate::core::train_config::TrainConfig;
use crate::optimizations::beijing_air_quality::cast::Optimizable;
use anyhow::{Context, Result as AnyhowResult};
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use fx_durable_ga::models::GenotypeManager;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};
use tokio::task;
use uuid::Uuid as GuidUuid;

const FEATURE_SOURCES: [Optimizable; 12] = [
    Optimizable::Pm25,
    Optimizable::Pm10,
    Optimizable::So2,
    Optimizable::No2,
    Optimizable::Co,
    Optimizable::O3,
    Optimizable::Temp,
    Optimizable::Pres,
    Optimizable::Dewp,
    Optimizable::Rain,
    Optimizable::Wspm,
    Optimizable::Wd,
];
const INTERPOLATION_WINDOWS: [usize; 3] = [1, 4, 16];
const NODE_WINDOWS: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
const HIDDEN_SIZES: [usize; 3] = [16, 32, 64];
const LEARNING_RATES: [f64; 5] = [1e-4, 4e-4, 7e-4, 1e-3, 2e-3];
const SEQUENCE_LENGTHS: [usize; 10] = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120];
const MIN_FEATURES: usize = 1;
const MAX_FEATURES: usize = FEATURE_SOURCES.len();
const MAX_NODES_PER_FEATURE: usize = 3;

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("Invalid genotype payload: {0}")]
    InvalidGenotype(String),
    #[error("Invalid user-defined config: {0}")]
    InvalidUserConfig(String),
    #[error("Worker subprocess failed: {0}")]
    WorkerFailed(String),
}

/// The genotype manager, registered to the GA genotype managers registry.
/// This struct performs mutation, crossover, randomization etc. of the Beijing genome.
pub struct BeijingGenotypeManager {
    model_save_path: Option<String>,
}

impl BeijingGenotypeManager {
    pub const TYPE_NAME: &'static str = "beijing_air_quality";
    pub const TYPE_HASH: i32 = fnv1a_hash_str_32(Self::TYPE_NAME) as i32;

    pub fn new(model_save_path: &str) -> Self {
        Self {
            model_save_path: Some(model_save_path.to_string()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct BeijingUserDefined {
    #[serde(default = "default_mutation_rate")]
    mutation_rate: f64,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_request_variables")]
    training: RequestVariables,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeijingPhenotype {
    features: Vec<Extract>,
    hidden_size: usize,
    learning_rate: f64,
    sequence_length: usize,
}

impl BeijingPhenotype {
    fn random(rng: &mut dyn RngCore) -> Self {
        let mut sources = FEATURE_SOURCES.to_vec();
        shuffle_vec(&mut sources, rng);
        let count = rng.random_range(MIN_FEATURES..=MAX_FEATURES);
        let mut features = Vec::with_capacity(count);
        for source in sources.into_iter().take(count) {
            features.push(random_feature(source, rng));
        }

        Self {
            features,
            hidden_size: sample_from_slice(&HIDDEN_SIZES, rng),
            learning_rate: sample_from_slice(&LEARNING_RATES, rng),
            sequence_length: sample_from_slice(&SEQUENCE_LENGTHS, rng),
        }
    }

    fn mutate(&mut self, rng: &mut dyn RngCore, params: MutationParams) {
        let rate = params.effective_rate();

        if rng.random::<f64>() < rate && self.features.len() > MIN_FEATURES {
            let idx = rng.random_range(0..self.features.len());
            self.features.remove(idx);
        }

        if rng.random::<f64>() < rate && self.features.len() < MAX_FEATURES {
            if let Some(source) = self.random_available_source(rng) {
                self.features.push(random_feature(source, rng));
            }
        }

        for feature in &mut self.features {
            if rng.random::<f64>() < rate {
                feature.interpolation = random_interpolation(rng);
            }
            if rng.random::<f64>() < rate {
                feature.nodes = random_nodes(rng);
            }
        }

        if rng.random::<f64>() < rate {
            self.hidden_size = mutate_discrete(self.hidden_size, &HIDDEN_SIZES, rng);
        }
        if rng.random::<f64>() < rate {
            self.learning_rate = mutate_discrete(self.learning_rate, &LEARNING_RATES, rng);
        }
        if rng.random::<f64>() < rate {
            self.sequence_length = mutate_discrete(self.sequence_length, &SEQUENCE_LENGTHS, rng);
        }

        self.ensure_constraints(rng);
    }

    fn crossover(
        parent1: &BeijingPhenotype,
        parent2: &BeijingPhenotype,
        rng: &mut dyn RngCore,
    ) -> Self {
        let mut features = Vec::new();

        for source in FEATURE_SOURCES {
            let f1 = parent1.feature_by_source(source);
            let f2 = parent2.feature_by_source(source);
            let selected = match (f1, f2) {
                (Some(a), Some(b)) => {
                    if rng.random::<f64>() < 0.5 {
                        Some(a.clone())
                    } else {
                        Some(b.clone())
                    }
                }
                (Some(a), None) => {
                    if rng.random::<f64>() < 0.6 {
                        Some(a.clone())
                    } else {
                        None
                    }
                }
                (None, Some(b)) => {
                    if rng.random::<f64>() < 0.6 {
                        Some(b.clone())
                    } else {
                        None
                    }
                }
                (None, None) => None,
            };

            if let Some(feature) = selected {
                if rng.random::<f64>() < 0.1 {
                    continue;
                }
                features.push(feature);
            }
        }

        if features.is_empty() {
            let source = sample_from_slice(&FEATURE_SOURCES, rng);
            features.push(random_feature(source, rng));
        }

        if features.len() > MAX_FEATURES {
            shuffle_vec(&mut features, rng);
            features.truncate(MAX_FEATURES);
        }

        let hidden_size =
            inherit_discrete(parent1.hidden_size, parent2.hidden_size, &HIDDEN_SIZES, rng);
        let learning_rate = inherit_discrete(
            parent1.learning_rate,
            parent2.learning_rate,
            &LEARNING_RATES,
            rng,
        );
        let sequence_length = inherit_discrete(
            parent1.sequence_length,
            parent2.sequence_length,
            &SEQUENCE_LENGTHS,
            rng,
        );

        Self {
            features,
            hidden_size,
            learning_rate,
            sequence_length,
        }
    }

    fn ensure_constraints(&mut self, rng: &mut dyn RngCore) {
        let mut seen = HashSet::new();
        self.features
            .retain(|extract| seen.insert(extract.source.clone()));

        if self.features.len() < MIN_FEATURES {
            if let Some(source) = self
                .random_available_source(rng)
                .or_else(|| FEATURE_SOURCES.first().copied())
            {
                self.features.push(random_feature(source, rng));
            }
        }
    }

    fn random_available_source(&self, rng: &mut dyn RngCore) -> Option<Optimizable> {
        let mut used = HashSet::new();
        for extract in &self.features {
            if let Ok(opt) = Optimizable::try_from(extract.source.as_str()) {
                used.insert(opt);
            }
        }
        let options: Vec<Optimizable> = FEATURE_SOURCES
            .iter()
            .copied()
            .filter(|source| !used.contains(source))
            .collect();
        choose_optional(&options, rng).copied()
    }

    fn feature_by_source(&self, source: Optimizable) -> Option<&Extract> {
        let name = source.to_string();
        self.features.iter().find(|extract| extract.source == name)
    }
}

struct MutationParams {
    mutation_rate: f64,
    temperature: f64,
    progress: f64,
}

impl MutationParams {
    fn effective_rate(&self) -> f64 {
        let cooling = (1.0 - self.progress).clamp(0.0, 1.0);
        let temp = self.temperature.clamp(0.0, 1.0);
        (self.mutation_rate * (0.5 + 0.5 * temp) * (0.25 + 0.75 * cooling)).clamp(0.01, 0.99)
    }
}

fn random_feature(source: Optimizable, rng: &mut dyn RngCore) -> Extract {
    let name = source.to_string();
    let mut extract = Extract::new(&name);
    extract.interpolation = random_interpolation(rng);
    extract.nodes = random_nodes(rng);
    extract
}

fn random_interpolation(rng: &mut dyn RngCore) -> Interpolation {
    let window = sample_from_slice(&INTERPOLATION_WINDOWS, rng);
    Interpolation::Linear(LinearInterpolator::new(window))
}

fn random_nodes(rng: &mut dyn RngCore) -> Vec<Node> {
    let count = rng.random_range(0..=MAX_NODES_PER_FEATURE);
    (0..count).map(|_| random_node(rng)).collect()
}

fn random_node(rng: &mut dyn RngCore) -> Node {
    match rng.random_range(0..4) {
        0 => Node::Std(Std::new(random_node_window(rng))),
        1 => Node::Ema(random_ema(rng)),
        2 => Node::Roc(Roc::new(random_node_window(rng))),
        _ => Node::ZScore(ZScore::new(random_node_window(rng))),
    }
}

fn random_node_window(rng: &mut dyn RngCore) -> usize {
    sample_from_slice(&NODE_WINDOWS, rng)
}

fn random_ema(rng: &mut dyn RngCore) -> Ema {
    let window = random_node_window(rng);
    let alpha = 2.0 / (window + 1) as f32;
    Ema::new(window, alpha)
}

fn mutate_discrete<T>(current: T, options: &[T], rng: &mut dyn RngCore) -> T
where
    T: Copy + PartialEq,
{
    let choices: Vec<T> = options
        .iter()
        .copied()
        .filter(|opt| *opt != current)
        .collect();
    if choices.is_empty() {
        current
    } else {
        let idx = rng.random_range(0..choices.len());
        choices[idx]
    }
}

fn inherit_discrete<T>(a: T, b: T, options: &[T], rng: &mut dyn RngCore) -> T
where
    T: Copy + PartialEq,
{
    if rng.random::<f64>() < 0.1 {
        return sample_from_slice(options, rng);
    }
    if rng.random::<f64>() < 0.5 { a } else { b }
}

fn sample_from_slice<T: Copy>(options: &[T], rng: &mut dyn RngCore) -> T {
    assert!(!options.is_empty(), "options must not be empty");
    let idx = rng.random_range(0..options.len());
    options[idx]
}

fn shuffle_vec<T>(items: &mut Vec<T>, rng: &mut dyn RngCore) {
    if items.len() < 2 {
        return;
    }
    for i in (1..items.len()).rev() {
        let j = rng.random_range(0..=i);
        items.swap(i, j);
    }
}

fn choose_optional<'a, T>(items: &'a [T], rng: &mut dyn RngCore) -> Option<&'a T> {
    if items.is_empty() {
        None
    } else {
        let idx = rng.random_range(0..items.len());
        Some(&items[idx])
    }
}

fn default_mutation_rate() -> f64 {
    0.1
}

fn default_temperature() -> f64 {
    0.35
}

fn default_request_variables() -> RequestVariables {
    RequestVariables::with_training_params(1, 25, 5, 10, 100)
}

fn decode_genotype(genotype: &Value) -> std::result::Result<BeijingPhenotype, EvaluationError> {
    serde_json::from_value(genotype.clone())
        .map_err(|err| EvaluationError::InvalidGenotype(err.to_string()))
}

fn decode_user_defined(value: &Value) -> std::result::Result<BeijingUserDefined, EvaluationError> {
    serde_json::from_value(value.clone())
        .map_err(|err| EvaluationError::InvalidUserConfig(err.to_string()))
}

fn append_time_transform(extracts: &mut Vec<Extract>, key: &str, period: u32) {
    extracts.push(Extract::new(key).with_node(Node::Sin(Sin::new(period as f32))));
    extracts.push(Extract::new(key).with_node(Node::Cos(Cos::new(period as f32))));
}

impl GenotypeManager for BeijingGenotypeManager {
    fn name(&self) -> &'static str {
        Self::TYPE_NAME
    }

    fn hash(&self) -> i32 {
        Self::TYPE_HASH
    }

    fn random(&self, rng: &mut dyn RngCore, _user_defined: &Value) -> AnyhowResult<Value> {
        let phenotype = BeijingPhenotype::random(rng);
        serde_json::to_value(&phenotype).context("Failed to serialize random Beijing genome")
    }

    fn mutate(
        &self,
        genotype: &mut Value,
        rng: &mut dyn RngCore,
        progress: f64,
        user_defined: &Value,
    ) -> AnyhowResult<()> {
        let mut phenotype = decode_genotype(genotype)?;
        let config = decode_user_defined(user_defined)?;

        let params = MutationParams {
            mutation_rate: config.mutation_rate,
            temperature: config.temperature,
            progress,
        };

        phenotype.mutate(rng, params);
        *genotype = serde_json::to_value(&phenotype)
            .context("Failed to serialize mutated Beijing genome")?;
        Ok(())
    }

    fn crossover(
        &self,
        parent1: &Value,
        parent2: &Value,
        rng: &mut dyn RngCore,
        _user_defined: &Value,
    ) -> AnyhowResult<Value> {
        let p1 = decode_genotype(parent1)?;
        let p2 = decode_genotype(parent2)?;
        let child = BeijingPhenotype::crossover(&p1, &p2, rng);
        serde_json::to_value(&child).context("Failed to serialize Beijing crossover result")
    }

    fn evaluate<'a>(
        &'a self,
        genotype: &'a Value,
        user_defined: &'a Value,
    ) -> BoxFuture<'a, AnyhowResult<f64>> {
        let genome = genotype.clone();
        let user_data = user_defined.clone();
        let model_save_path = self.model_save_path.clone();

        Box::pin(async move {
            let phenotype = decode_genotype(&genome)?;
            let config = decode_user_defined(&user_data)?;

            let mut features = phenotype.features.clone();
            features.push(Extract::new("pos_x"));
            features.push(Extract::new("pos_y"));
            append_time_transform(&mut features, "month", 12);
            append_time_transform(&mut features, "day_of_week", 7);
            append_time_transform(&mut features, "hour", 24);

            let mut targets = Vec::with_capacity(1);
            targets.push(
                Extract::new("TEMP")
                    .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            );

            let train_config = TrainConfig::new(
                phenotype.hidden_size,
                phenotype.sequence_length,
                config.training.prediction_horizon,
                features,
                targets,
                config.training.epochs,
                config.training.batch_size,
                phenotype.learning_rate,
            )?
            .with_patience(config.training.patience)?
            .with_validation_start_epoch(config.training.validation_start_epoch)?;

            let genotype_id = GuidUuid::now_v7();
            let model_path = model_save_path
                .as_ref()
                .map(|path| format!("{}/{}", path, genotype_id));

            let train_request = Request {
                genotype_id,
                train_config: serde_json::to_value(&train_config)?,
                model_save_path: model_path,
            };

            if let Some(path) = train_request
                .model_save_path
                .as_ref()
                .map(|p| format!("{}.request.json", p))
            {
                train_request.save(&path)?;
            }

            tracing::info!(
                message = "Spawning training binary",
                genotype_id = %train_request.genotype_id
            );

            let json_request =
                serde_json::to_string(&train_request).context("Failed to encode train request")?;

            let task = task::spawn_blocking(move || -> AnyhowResult<Response> {
                let mut child = Command::new("./target/release/beijing")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|err| {
                        EvaluationError::WorkerFailed(format!("Failed to spawn binary: {err}"))
                    })?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(json_request.as_bytes()).map_err(|err| {
                        EvaluationError::WorkerFailed(format!("Failed to write to binary: {err}"))
                    })?;
                } else {
                    return Err(EvaluationError::WorkerFailed(
                        "Failed to connect to binary stdin".to_string(),
                    )
                    .into());
                }

                let output = child.wait_with_output().map_err(|err| {
                    EvaluationError::WorkerFailed(format!("Failed to wait for binary: {err}"))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(EvaluationError::WorkerFailed(format!(
                        "Binary failed with status {}: {}",
                        output.status, stderr
                    ))
                    .into());
                }

                serde_json::from_slice::<Response>(&output.stdout).map_err(|err| {
                    EvaluationError::WorkerFailed(format!("Failed to parse binary response: {err}"))
                        .into()
                })
            });

            let response = task.await.map_err(|err| {
                EvaluationError::WorkerFailed(format!("Task failed before completion: {err}"))
            })??;

            tracing::info!(
                message = "Training completed",
                genotype_id = %response.genotype_id,
                fitness = response.fitness
            );

            Ok(response.fitness)
        })
    }
}
