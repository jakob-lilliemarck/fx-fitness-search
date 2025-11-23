use fx_durable_ga_app::core::{ingestion::Extract, train_config::TrainConfig};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use uuid::Uuid;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
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

async fn run() -> anyhow::Result<()> {
    // Read request from stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let request: Request = serde_json::from_str(&buffer)?;

    tracing::info!(
        message = "Training started",
        dataset = "Beijing air quality",
        genotype_id = ?request.genotype_id,
    );

    // Features
    let mut pipelines_f: Vec<Extract> = Vec::new();

    // Targets
    let mut pipelines_t: Vec<Extract> = Vec::new();

    Ok(())
}
