use fx_durable_ga_app::config::{App, ServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename("src/bin/.env.server").ok();

    let conf = ServerConfig::from_env()?;

    let server = App::server(conf).await?;

    tokio::signal::ctrl_c().await?;

    server.stop().await;

    Ok(())
}
