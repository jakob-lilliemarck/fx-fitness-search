use crate::optimizations::beijing_air_quality::new_beijing_service;
use sqlx::{PgPool, types::Uuid};
use std::{sync::Arc, time::Duration};
use tokio::task::JoinSet;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing environment variable \"{key}\"\n\tMessage: {message}")]
    Missing { key: String, message: String },
    #[error("Could not parse environment variable: {key}\n\tGot: {value}\n\tMessage: {message}")]
    Invalid {
        key: String,
        value: String,
        message: String,
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
pub struct ShutdownTimeoutSeconds;
pub struct ModelSavePath;
pub struct MaxWorkers;
pub struct BatchSize;
pub struct HttpAddr;
pub struct ApiBaseUrl;

impl Var for DatabaseUrl {
    const NAME: &'static str = "DATABASE_URL";
    type Type = String;

    fn from_env() -> Result<String, ConfigError> {
        let database_url = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        Ok(database_url)
    }
}

impl Var for HostId {
    const NAME: &str = "HOST_ID";
    type Type = Uuid;

    fn from_env() -> Result<Uuid, ConfigError> {
        let host_id_str = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        Uuid::parse_str(&host_id_str).map_err(|err| ConfigError::Invalid {
            key: Self::NAME.to_string(),
            value: host_id_str,
            message: err.to_string(),
        })
    }
}

impl Var for LeaseSeconds {
    const NAME: &str = "LEASE_SECONDS";
    type Type = Duration;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let lease_seconds_str = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        let lease_seconds =
            lease_seconds_str
                .parse::<u64>()
                .map_err(|err| ConfigError::Invalid {
                    key: Self::NAME.to_string(),
                    value: lease_seconds_str,
                    message: err.to_string(),
                })?;
        Ok(Duration::from_secs(lease_seconds))
    }
}

impl Var for ShutdownTimeoutSeconds {
    const NAME: &str = "SHUTDOWN_TIMEOUT_SECONDS";
    type Type = Duration;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let shutdown_timeout_str =
            std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
                key: Self::NAME.to_string(),
                message: err.to_string(),
            })?;

        let shutdown_timeout =
            shutdown_timeout_str
                .parse::<u64>()
                .map_err(|err| ConfigError::Invalid {
                    key: Self::NAME.to_string(),
                    value: shutdown_timeout_str,
                    message: err.to_string(),
                })?;
        Ok(Duration::from_secs(shutdown_timeout))
    }
}

impl Var for ModelSavePath {
    const NAME: &str = "MODEL_SAVE_PATH";
    type Type = String;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let path = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;
        Ok(path)
    }
}

impl Var for MaxWorkers {
    const NAME: &str = "MAX_WORKERS";
    type Type = Option<usize>;

    fn from_env() -> Result<Self::Type, ConfigError> {
        match std::env::var(Self::NAME) {
            Ok(val) => {
                let n = val.parse::<usize>().map_err(|err| ConfigError::Invalid {
                    key: Self::NAME.to_string(),
                    value: val.clone(),
                    message: err.to_string(),
                })?;
                if n < 1 {
                    return Err(ConfigError::Invalid {
                        key: Self::NAME.to_string(),
                        value: val,
                        message: "must be >= 1".to_string(),
                    });
                }
                Ok(Some(n))
            }
            Err(_) => Ok(None),
        }
    }
}

impl Var for BatchSize {
    const NAME: &str = "BATCH_SIZE";
    type Type = usize;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let val = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        val.parse::<usize>().map_err(|err| ConfigError::Invalid {
            key: Self::NAME.to_string(),
            value: val,
            message: err.to_string(),
        })
    }
}

impl Var for HttpAddr {
    const NAME: &str = "HTTP_ADDR";
    type Type = std::net::SocketAddr;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let val = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        val.parse::<std::net::SocketAddr>()
            .map_err(|err| ConfigError::Invalid {
                key: Self::NAME.to_string(),
                value: val,
                message: err.to_string(),
            })
    }
}

impl Var for ApiBaseUrl {
    const NAME: &str = "API_BASE_URL";
    type Type = String;

    fn from_env() -> Result<Self::Type, ConfigError> {
        let val = std::env::var(Self::NAME).map_err(|err| ConfigError::Missing {
            key: Self::NAME.to_string(),
            message: err.to_string(),
        })?;

        Ok(val)
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub database_url: String,
    pub host_id: Uuid,
    pub lease_seconds: Duration,
    pub shutdown_timeout_seconds: Duration,
    pub model_save_path: String,
    pub workers: usize,
    pub batch_size: usize,
    pub http_addr: std::net::SocketAddr,
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub database_url: String,
    pub host_id: Uuid,
    pub model_save_path: String,
    pub api_base_url: String,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let database_url = DatabaseUrl::from_env()?;

        let host_id = HostId::from_env()?;

        let lease_seconds = LeaseSeconds::from_env()?;

        let shutdown_timeout_seconds = ShutdownTimeoutSeconds::from_env()?;

        let model_save_path = ModelSavePath::from_env()?;

        let batch_size = BatchSize::from_env()?;
        let http_addr = HttpAddr::from_env()?;

        // Determine worker count based on backend.
        let max_workers = MaxWorkers::from_env()?;

        #[cfg(feature = "backend-ndarray")]
        let workers = {
            // For CPU backend: default to half physical cores, capped by MAX_WORKERS if set
            let physical_cores = num_cpus::get_physical();
            let default_workers = physical_cores / 2;
            max_workers.map_or(default_workers, |max| default_workers.min(max))
        };

        #[cfg(not(feature = "backend-ndarray"))]
        let workers = {
            // For GPU backend: use MAX_WORKERS directly, error if not set
            max_workers.ok_or_else(|| ConfigError::Missing {
                key: MaxWorkers::NAME.to_string(),
                message: "MAX_WORKERS is required when using GPU backend".to_string(),
            })?
        };

        tracing::info!(
            message = "Configuration loaded",
            database_url = %database_url,
            host_id = %host_id,
            lease_seconds = lease_seconds.as_secs(),
            shutdown_timeout_seconds = shutdown_timeout_seconds.as_secs(),
            model_save_path = %model_save_path,
            workers = workers
        );

        Ok(ServerConfig {
            database_url,
            host_id,
            lease_seconds,
            shutdown_timeout_seconds,
            model_save_path,
            workers,
            batch_size,
            http_addr,
        })
    }
}

impl ClientConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let database_url = DatabaseUrl::from_env()?;
        let host_id = HostId::from_env()?;
        let model_save_path = ModelSavePath::from_env()?;
        let api_base_url = ApiBaseUrl::from_env()?;

        tracing::info!(
            message = "Configuration loaded",
            database_url = %database_url,
            model_save_path = %model_save_path
        );

        Ok(ClientConfig {
            database_url,
            host_id,
            model_save_path,
            api_base_url,
        })
    }
}

#[derive(Debug)]
pub enum Conf {
    Client(ClientConfig),
    Server(ServerConfig),
}

impl Conf {
    pub fn database_url(&self) -> &str {
        match &self {
            Conf::Client(ClientConfig { database_url, .. }) => database_url,
            Conf::Server(ServerConfig { database_url, .. }) => database_url,
        }
    }

    pub fn model_save_path(&self) -> &str {
        match &self {
            Conf::Client(ClientConfig {
                model_save_path, ..
            }) => model_save_path,
            Conf::Server(ServerConfig {
                model_save_path, ..
            }) => model_save_path,
        }
    }
}

pub struct Tasks {
    cancel: tokio_util::sync::CancellationToken,
    tasks: JoinSet<Result<(), anyhow::Error>>,
}

impl Tasks {
    pub fn start(
        mut event_listener: fx_event_bus::Listener,
        mut jobs_listener: fx_mq_jobs::Listener,
        shutdown_timeout: Duration,
    ) -> Self {
        let mut set = JoinSet::new();
        let cancel = tokio_util::sync::CancellationToken::new();
        let cancel_jobs = cancel.clone();
        let cancel_events = cancel.clone();

        set.spawn(async move {
            tokio::select! {
                res = event_listener.listen(None) => res.map_err(|e| anyhow::anyhow!(e)),
                _ = cancel_events.cancelled() => {
                    tracing::info!("Event listener received shutdown signal");
                    Ok(())
                }
            }
        });

        set.spawn(async move {
            tokio::select! {
                res = jobs_listener.listen() => res.map_err(|e| anyhow::anyhow!(e)),
                _ = cancel_jobs.cancelled() => {
                    tracing::info!("Job listener received shutdown signal");
                    jobs_listener.stop(shutdown_timeout).await.map_err(|e| anyhow::anyhow!(e))
                }
            }
        });

        Self { cancel, tasks: set }
    }

    pub async fn stop(mut self) {
        self.cancel.cancel();

        while let Some(result) = self.tasks.join_next().await {
            match result {
                Ok(result) => {
                    if let Err(err) = result {
                        tracing::error!(
                            message = "An error occured in task",
                            error = ?err
                        )
                    }
                }
                Err(err) => {
                    tracing::error!(message="Failed to join task", error=?err)
                }
            }
        }
    }
}

pub struct GADependencies {
    host_id: Uuid,
    pool: PgPool,
}

impl fx_durable_ga::repositories::PoolProvider for GADependencies {
    fn get(&self) -> &PgPool {
        &self.pool
    }
}

impl fx_durable_ga::repositories::encoders::Dependencies for GADependencies {
    fn capacity(&self) -> usize {
        3
    }

    fn ttl(&self) -> Duration {
        Duration::from_secs(60 * 60)
    }
}

impl fx_durable_ga::services::optimization::Dependencies for GADependencies {
    fn host_id(&self) -> Uuid {
        self.host_id
    }

    fn max_deduplication_attempts(&self) -> i32 {
        5
    }
}

pub struct App {
    pool: PgPool,
    ga: Arc<fx_durable_ga::bootstrap::App>,
    tasks: Option<Tasks>,
}

impl App {
    pub async fn client(conf: ClientConfig) -> anyhow::Result<Self> {
        // Create a database connection pool
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .connect(&conf.database_url)
            .await?;

        let ga_dependencies = GADependencies {
            host_id: conf.host_id,
            pool: pool.clone(),
        };

        let ga = Arc::new(
            fx_durable_ga::bootstrap::App::builder(&ga_dependencies)
                .with_service(new_beijing_service(&conf.model_save_path, 0))?
                .build()?,
        );

        // Run required migrations migrations
        fx_durable_ga::migrations::run_default_migrations(&pool).await?;

        Ok(Self {
            pool,
            ga,
            tasks: None,
        })
    }

    pub async fn server(conf: ServerConfig) -> anyhow::Result<Self> {
        // Create a database connection pool
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(10)
            .connect(&conf.database_url)
            .await?;

        let ga_dependencies = GADependencies {
            host_id: conf.host_id,
            pool: pool.clone(),
        };

        let ga = Arc::new(
            fx_durable_ga::bootstrap::App::builder(&ga_dependencies)
                .with_service(new_beijing_service(&conf.model_save_path, conf.batch_size))?
                .build()?,
        );

        let http_addr = conf.http_addr;
        let ga_http = ga.clone();
        tokio::spawn(async move {
            tracing::info!("swagger docs: http://{}/docs", http_addr);
            tracing::info!("lineage page: http://{}/lineage", http_addr);

            if let Err(err) = ga_http.serve_http(http_addr).await {
                tracing::error!(message = "http server failed", error = ?err);
            }
        });

        // Run required migrations migrations
        fx_durable_ga::migrations::run_default_migrations(&pool).await?;

        let mut app = Self {
            pool,
            ga,
            tasks: None,
        };

        // --- Event listener setup ---
        let mut registry = fx_event_bus::EventHandlerRegistry::new();
        app.ga.register_event_handlers(&mut registry);
        let event_listener = fx_event_bus::Listener::new(app.pool.clone(), registry);

        // --- Job listener setup ---
        let jobs_listener = fx_mq_jobs::Listener::new(
            app.pool.clone(),
            app.ga
                .register_job_handler(fx_mq_jobs::RegistryBuilder::new()),
            conf.workers,
            conf.host_id,
            conf.lease_seconds,
        )
        .await?;

        app.tasks = Some(Tasks::start(
            event_listener,
            jobs_listener,
            conf.shutdown_timeout_seconds,
        ));

        Ok(app)
    }

    pub fn ga(&self) -> Arc<fx_durable_ga::bootstrap::App> {
        self.ga.clone()
    }

    pub async fn stop(self) {
        if let Some(tasks) = self.tasks {
            tasks.stop().await;
        }
    }
}
