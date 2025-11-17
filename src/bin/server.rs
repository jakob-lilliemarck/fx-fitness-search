use fx_durable_ga_app::config::{App, ServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename(".env.shared").ok();
    dotenv::from_filename(".env.server").ok();

    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::WARN)
        .init();

    let conf = ServerConfig::from_env()?;

    tracing::info!(message = "starting server", conf = ?conf);

    let server = App::server(conf).await?;

    tokio::signal::ctrl_c().await?;

    server.stop().await;

    Ok(())
}
