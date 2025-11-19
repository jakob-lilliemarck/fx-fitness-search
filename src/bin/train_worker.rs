use burn::data::dataloader::Dataset;
use fx_durable_ga_app::core::model::{FeedForward, SequenceModel};
use fx_durable_ga_app::core::preprocessor::Transform;
use fx_durable_ga_app::core::train_config::TrainConfig;
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use fx_durable_ga_app::optimizations::beijing_air_quality::ingestion::build_dataset_from_file;
#[derive(Debug, Deserialize)]
struct TrainRequest {
    /// The configuration for this training run
    train_config: TrainConfig,
    /// The UUID of this genotype
    genotype_id: String,
    /// List of CSV file paths to load data from
    data_paths: Vec<String>,
    /// Features as serialized strings (e.g., "feat_0=TEMP:ZSCORE(48)")
    features: Vec<String>,
    /// Targets as serialized strings (e.g., "target_temp=TEMP")
    targets: Vec<String>,
    /// Training hyperparameters
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

/// Response from training - written to stdout
#[derive(Debug, Serialize)]
struct TrainResponse {
    /// The fitness value (validation loss)
    fitness: f64,
    /// The genotype ID (echoed back)
    genotype_id: String,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    // Read request from stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let request: TrainRequest = serde_json::from_str(&buffer)?;

    tracing::info!(
        message = "Training worker started",
        genotype_id = &request.genotype_id,
        num_files = request.data_paths.len(),
    );

    // Parse features from strings
    let features: Vec<Transform> = request
        .features
        .iter()
        .map(|s| {
            s.as_str()
                .try_into()
                .map_err(|e: fx_durable_ga_app::core::preprocessor::Error| e.into())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let targets: anyhow::Result<Vec<Transform>> = request
        .targets
        .iter()
        .map(|s| {
            s.as_str()
                .try_into()
                .map_err(|e: fx_durable_ga_app::core::preprocessor::Error| e.into())
        })
        .collect();
    let targets = targets?;

    // Build datasets from all files
    let mut all_train_items = Vec::new();
    let mut all_valid_items = Vec::new();

    for path in &request.data_paths {
        tracing::info!(message = "Loading file", path = path);

        let builder = build_dataset_from_file(path, &features, &targets)?;

        let (train_dataset, valid_dataset) = builder.build(
            request.train_config.sequence_length,
            request.train_config.prediction_horizon,
            Some(0.8),
        )?;

        // Extract items with metadata
        for i in 0..train_dataset.len() {
            if let (Some(item), Some(metadata)) =
                (train_dataset.get(i), train_dataset.get_metadata(i))
            {
                all_train_items.push((metadata.clone(), item));
            }
        }

        if let Some(valid) = valid_dataset {
            for i in 0..valid.len() {
                if let (Some(item), Some(metadata)) = (valid.get(i), valid.get_metadata(i)) {
                    all_valid_items.push((metadata.clone(), item));
                }
            }
        }
    }

    tracing::info!(
        message = "Datasets loaded",
        train_items = all_train_items.len(),
        valid_items = all_valid_items.len(),
    );

    // Create unified datasets
    let train_dataset =
        fx_durable_ga_app::core::dataset::SequenceDataset::from_items(all_train_items);
    let valid_dataset =
        fx_durable_ga_app::core::dataset::SequenceDataset::from_items(all_valid_items);

    // Set up training environment
    type Backend = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::default();

    // Create model
    let model = FeedForward::<Backend>::new(
        &device,
        request.train_config.input_size,
        request.train_config.hidden_size,
        request.train_config.output_size,
        request.train_config.sequence_length,
    );

    tracing::info!(message = "Model created");

    // Train the model (sync)
    let (_trained_model, best_valid_loss) = fx_durable_ga_app::core::train::train_sync(
        &device,
        &train_dataset,
        &valid_dataset,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        model,
        None,
        None,
    );

    tracing::info!(
        message = "Training complete",
        best_valid_loss = best_valid_loss,
    );

    // Write response to stdout
    let response = TrainResponse {
        fitness: best_valid_loss as f64,
        genotype_id: request.genotype_id,
    };

    let json = serde_json::to_string(&response)?;
    io::stdout().write_all(json.as_bytes())?;
    io::stdout().flush()?;

    Ok(())
}

