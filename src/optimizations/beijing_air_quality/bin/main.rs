use fx_durable_ga_app::core::dataset::ManySequencesAdapter;
use fx_durable_ga_app::core::interpolation::LinearInterpolator;
use fx_durable_ga_app::core::model::{FeedForward, SequenceModel};
use fx_durable_ga_app::core::preprocessor::{Cos, Node, Sin};
use fx_durable_ga_app::core::{
    ingestion::Extract, interpolation::Interpolation, train_config::TrainConfig,
};
use fx_durable_ga_app::optimizations::beijing_air_quality::ingest;
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use uuid::Uuid;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

#[derive(Debug, Deserialize)]
struct Request {
    genotype_id: Uuid,
    train_config: TrainConfig,
    processors_features: Vec<Extract>,
    processors_targets: Vec<Extract>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

#[derive(Debug, Serialize)]
struct Response {
    fitness: f64,
    genotype_id: Uuid,
}

fn append_time_transform(extracts: &mut Vec<Extract>, key: &str, period: u32) {
    extracts.push(Extract::new(key).with_node(Node::Sin(Sin::new(period as f32))));
    extracts.push(Extract::new(key).with_node(Node::Cos(Cos::new(period as f32))));
}

async fn run() -> anyhow::Result<()> {
    // Read request from stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let mut request: Request = serde_json::from_str(&buffer)?;

    tracing::info!(
        message = "Training started",
        dataset = "Beijing air quality",
        genotype_id = ?request.genotype_id,
    );

    // Add processors that should _always_ be used
    append_time_transform(&mut request.processors_features, "month", 12);
    append_time_transform(&mut request.processors_features, "day_of_week", 7);
    append_time_transform(&mut request.processors_features, "hour", 24);

    request.processors_targets.push(
        Extract::new("TEMP").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
    );

    type Backend = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let model = FeedForward::<Backend>::new(
        &device,
        request.processors_features.len(),
        request.train_config.hidden_size,
        request.processors_targets.len(),
        request.train_config.sequence_length,
    );

    let dataset = ingest(request.processors_features, request.processors_targets).await?;

    let (sequences_training, sequences_validation) = dataset.split(0.8)?;

    let dataset_training = ManySequencesAdapter::new(
        sequences_training,
        request.train_config.prediction_horizon,
        request.train_config.sequence_length,
    );

    let dataset_validation = ManySequencesAdapter::new(
        sequences_validation,
        request.train_config.prediction_horizon,
        request.train_config.sequence_length,
    );

    // Train the model (sync)
    let (_trained_model, best_valid_loss) = fx_durable_ga_app::core::train::train_sync(
        &device,
        &dataset_training,
        &dataset_validation,
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
    let response = Response {
        fitness: best_valid_loss as f64,
        genotype_id: request.genotype_id,
    };

    let json = serde_json::to_string(&response)?;
    io::stdout().write_all(json.as_bytes())?;
    io::stdout().flush()?;

    Ok(())
}
