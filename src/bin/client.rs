use clap::{Parser, Subcommand};
use fx_durable_ga_app::config::{App, ClientConfig};
use fx_durable_ga_app::optimizations::beijing_air_quality::BeijingCommand;

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

    let client = App::client(conf).await?;

    let args = Args::parse();

    match args.command {
        Command::Beijing(cmd) => {
            cmd.execute(client.get_svc()).await?;
        }
    }

    Ok(())
}
