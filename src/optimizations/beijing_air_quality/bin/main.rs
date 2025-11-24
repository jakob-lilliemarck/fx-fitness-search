use fx_durable_ga_app::core::dataset::ManySequencesAdapter;
use fx_durable_ga_app::core::interpolation::LinearInterpolator;
use fx_durable_ga_app::core::model::{FeedForward, SequenceModel};
use fx_durable_ga_app::core::preprocessor::{Cos, Node, Sin};
use fx_durable_ga_app::core::{ingestion::Extract, interpolation::Interpolation};
use fx_durable_ga_app::optimizations::beijing_air_quality::{
    ingest,
    protocol::{Request, Response},
};
use std::io::{self, Read, Write};

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
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

    // ============================================================
    // This section hardcodes some features and targets, to take
    // them out of the optimization - they're effectivly constants.
    // ============================================================
    //
    // Add features that should _always_ be present
    append_time_transform(&mut request.train_config.features, "month", 12);
    append_time_transform(&mut request.train_config.features, "day_of_week", 7);
    append_time_transform(&mut request.train_config.features, "hour", 24);

    // Add targets that should _always_ be present
    request.train_config.targets.push(
        Extract::new("TEMP").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
    );
    // ============================================================

    type Backend = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let model = FeedForward::<Backend>::new(
        &device,
        request.train_config.features.len(),
        request.train_config.hidden_size,
        request.train_config.targets.len(),
        request.train_config.sequence_length,
    );

    let dataset = ingest(
        &request.train_config.features,
        &request.train_config.targets,
    )
    .await?;

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
        request.model_save_path,
        Some(request.train_config),
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
