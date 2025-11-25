use fx_durable_ga_app::core::dataset::ManySequencesAdapter;
use fx_durable_ga_app::core::ingestion::Extract;
use fx_durable_ga_app::core::model::{FeedForward, SequenceModel};
use fx_durable_ga_app::optimizations::beijing_air_quality::{
    ingest,
    protocol::{Request, Response},
};
use std::io::{self, Read, Write};
use tracing::{Level, instrument};

#[tokio::main]
async fn main() {
    // Log diagnostics to stderr
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    if let Err(e) = async {
        // Parse the input json document
        let request = parse()?;

        // Create datasets
        let (dataset_training, dataset_validation) = prepare(
            &request.train_config.features,
            &request.train_config.targets,
            request.train_config.prediction_horizon,
            request.train_config.sequence_length,
        )
        .await?;

        // Train the model
        let response = train(request, &dataset_training, &dataset_validation).await?;

        // Flush response to stdout
        let json = serde_json::to_string(&response)?;
        io::stdout().write_all(json.as_bytes())?;
        io::stdout().flush()?;

        Ok::<_, anyhow::Error>(())
    }
    .await
    {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

#[instrument(level = "info")]
fn parse() -> anyhow::Result<Request> {
    // Read Request from stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    serde_json::from_str(&buffer).map_err(Into::into)
}

#[instrument(level = "info")]
async fn prepare(
    features: &[Extract],
    targets: &[Extract],
    prediction_horizon: usize,
    sequence_length: usize,
) -> anyhow::Result<(ManySequencesAdapter, ManySequencesAdapter)> {
    let dataset = ingest(&features, &targets).await?;

    // Split each sequence at a factor.
    //
    // NOTE:
    // Each sequence is split temporally, but if sequences are of different lengths
    // that could cause some data leakage, leaking some times from the validation set of one
    // sequence into the training set of another.
    let (sequences_training, sequences_validation) = dataset.split(0.8)?;

    // Create the training dataset
    let training =
        ManySequencesAdapter::new(sequences_training, prediction_horizon, sequence_length);

    // Create the validation dataset
    let validation =
        ManySequencesAdapter::new(sequences_validation, prediction_horizon, sequence_length);

    Ok((training, validation))
}

#[instrument(level = "info", skip(request, training, validation))]
async fn train(
    request: Request,
    training: &ManySequencesAdapter,
    validation: &ManySequencesAdapter,
) -> anyhow::Result<Response> {
    type Backend = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let model = FeedForward::<Backend>::new(
        &device,
        request.train_config.features.len(),
        request.train_config.hidden_size,
        request.train_config.targets.len(),
        request.train_config.sequence_length,
    );

    let (_trained_model, best_valid_loss) = fx_durable_ga_app::core::train::train_sync(
        &device,
        training,
        validation,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        model,
        request.model_save_path,
        Some(request.train_config),
    );

    tracing::info!("Training complete");

    Ok(Response {
        fitness: best_valid_loss as f64,
        genotype_id: request.genotype_id,
    })
}
