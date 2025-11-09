use sqlx::{PgPool, types::Uuid};
use std::{env::VarError, error::Error, sync::Arc, time::Duration};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing environment variable: {0}")]
    Missing(#[from] VarError),
    #[error("Could not parse environment variable for {key}, got: {value}")]
    Invalid {
        key: String,
        value: String,
        error: Box<dyn Error + Send + Sync>,
    },
}

pub trait Var {
    const NAME: &'static str;
    type Type;

    fn from_env() -> Result<Self::Type, ConfigError>;
}

pub struct DatabaseUrl;
pub struct HostId;
pub struct LeaseSeconds;

impl Var for DatabaseUrl {
    const NAME: &'static str = "DATABASE_URL";
    type Type = String;

    fn from_env() -> Result<String, ConfigError> {
        let database_url = std::env::var(Self::NAME)?;
        Ok(database_url)
    }
}

impl Var for HostId {
    const NAME: &str = "HOST_ID";
    type Type = Uuid;

    fn from_env() -> Result<Uuid, ConfigError> {
        let host_id_str = std::env::var(Self::NAME)?;
        Uuid::parse_str(&host_id_str).map_err(|err| ConfigError::Invalid {
            key: Self::NAME.to_string(),
            value: host_id_str,
            error: Box::new(err),
        })
    }
}

impl Var for LeaseSeconds {
    const NAME: &str = "LEASE_SECONDS";
    type Type = Duration;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let lease_seconds_str = std::env::var(Self::NAME)?;
        let lease_seconds =
            lease_seconds_str
                .parse::<u64>()
                .map_err(|err| ConfigError::Invalid {
                    key: Self::NAME.to_string(),
                    value: lease_seconds_str,
                    error: Box::new(err),
                })?;
        Ok(Duration::from_secs(lease_seconds))
    }
}

pub struct ServerConfig {
    pub database_url: String,
    pub host_id: Uuid,
    pub lease_seconds: Duration,
    pub workers: usize,
}

pub struct ClientConfig {
    pub database_url: String,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let database_url = DatabaseUrl::from_env()?;

        let host_id = HostId::from_env()?;

        let lease_seconds = LeaseSeconds::from_env()?;

        // Get the number of physical cores on the current machine.
        // At the time of writing this I know that my current training with NdArray runs on 2 cores.
        // As such we divide by two and round down, to get the numbers of worker to run.
        // Note that this couples the logic of this server to knowledge of the training,
        // which obviously is less than ideal - however, this is just an experiment.
        let physical_cores = num_cpus::get_physical();
        let workers = physical_cores / 2;

        Ok(ServerConfig {
            database_url,
            host_id,
            lease_seconds,
            workers,
        })
    }
}

impl ClientConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let database_url = DatabaseUrl::from_env()?;

        Ok(ClientConfig { database_url })
    }
}

pub async fn bootstrap(
    database_url: &str,
) -> anyhow::Result<(Arc<fx_durable_ga::optimization::Service>, PgPool)> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    // Create a database connection pool
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await?;

    // Run required migrations migrations
    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_jobs::run_migrations(&pool, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME).await?;
    fx_durable_ga::run_migrations(&pool).await?;

    // Create the GA service and wrap it in an Arc
    let service = Arc::new(fx_durable_ga::bootstrap(pool.clone()).await?.build());

    return Ok((service, pool));
}
