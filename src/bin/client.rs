use clap::{Parser, Subcommand};
use fx_durable_ga_app::config::{App, ClientConfig};
use fx_durable_ga_app::optimizations::beijing_air_quality::BeijingCommand;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "fx-durable-ga-app",
    version = "1.0",
    author = "Jakob",
    about = "Distributed optimizations"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Beijing air quality domain-specific operations
    #[command(subcommand)]
    Beijing(BeijingCommand),
    /// Interrupt a running optimization request
    Interrupt {
        /// The request ID to interrupt
        #[arg(long, required = true)]
        request_id: uuid::Uuid,
    },
    /// Backfill missing genotype embeddings via HTTP
    BackfillGenotypeEmbeddings {
        /// Optional request ID filter
        #[arg(long)]
        request_id: Option<uuid::Uuid>,
        /// Optional generation ID filters (repeatable)
        #[arg(long, value_delimiter = ' ')]
        generation_id: Vec<i32>,
        /// Optional genotype ID filters (repeatable)
        #[arg(long, value_delimiter = ' ')]
        genotype_id: Vec<uuid::Uuid>,
        /// Optional evaluation filter
        #[arg(long)]
        has_evaluation: Option<bool>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename(".env.shared").ok();
    dotenv::from_filename(".env.client").ok();

    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let conf = ClientConfig::from_env()?;
    let api_base_url = conf.api_base_url.clone();

    let client = App::client(conf).await?;

    let args = Args::parse();

    match args.command {
        Command::Beijing(cmd) => {
            cmd.execute(client.ga()).await?;
        }
        Command::Interrupt { request_id } => {
            client
                .ga()
                .services()
                .optimization()
                .request_interrupt(request_id)
                .await?;
            tracing::info!("Interrupted optimization request: {}", request_id);
        }
        Command::BackfillGenotypeEmbeddings {
            request_id,
            generation_id,
            genotype_id,
            has_evaluation,
        } => {
            let mut payload = serde_json::Map::new();

            if let Some(request_id) = request_id {
                payload.insert("request_id".to_string(), json!(request_id));
            }

            if !generation_id.is_empty() {
                let values = generation_id
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>();
                payload.insert("generation_ids".to_string(), json!(values));
            }

            if !genotype_id.is_empty() {
                payload.insert("genotype_ids".to_string(), json!(genotype_id));
            }

            if let Some(has_evaluation) = has_evaluation {
                payload.insert("has_evaluation".to_string(), json!(has_evaluation));
            }

            let url = format!(
                "{}/genotypes/embeddings",
                api_base_url.trim_end_matches('/')
            );
            let response = reqwest::Client::new()
                .post(url)
                .json(&payload)
                .send()
                .await?;

            if response.status() != reqwest::StatusCode::ACCEPTED {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("backfill request failed: {} - {}", status, body);
            }

            tracing::info!("Backfill request accepted");
        }
    }

    Ok(())
}
