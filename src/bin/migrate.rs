use fx_durable_ga::migrations::run_default_migrations;
use sqlx::postgres::PgPoolOptions;
use tracing::Level;

// Use this binary to run all required migrations locally.
// Failing to run migrations will cause compilation issues as sqlx uses the schema to assert typing in queries
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename(".env").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    run_default_migrations(&pool).await?;

    Ok(())
}
