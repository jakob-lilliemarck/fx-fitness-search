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

#[derive(Clone)]
pub struct ServerConfig {
    pub database_url: String,
    pub host_id: Uuid,
    pub lease_seconds: Duration,
    pub workers: usize,
}

#[derive(Clone)]
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

pub enum Conf {
    Client(ClientConfig),
    Server(ServerConfig),
}

pub struct Tasks {
    cancel: tokio_util::sync::CancellationToken,
    tasks: JoinSet<Result<(), anyhow::Error>>,
}

impl Tasks {
    pub fn start(
        mut event_listener: fx_event_bus::Listener,
        mut jobs_listener: fx_mq_jobs::Listener,
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
                    jobs_listener.stop(Duration::from_secs(10)).await.map_err(|e| anyhow::anyhow!(e))
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

pub struct App {
    pool: PgPool,
    svc: Arc<fx_durable_ga::optimization::Service>,
    tasks: Option<Tasks>,
}

impl App {
    pub async fn client(conf: ClientConfig) -> anyhow::Result<Self> {
        Self::new(Conf::Client(conf)).await
    }

    pub async fn server(conf: ServerConfig) -> anyhow::Result<Self> {
        let mut app = Self::new(Conf::Server(conf.clone())).await?;

        // --- Event listener setup ---
        let mut registry = fx_event_bus::EventHandlerRegistry::new();
        fx_durable_ga::register_event_handlers(
            std::sync::Arc::new(fx_mq_jobs::Queries::new(fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME)),
            app.svc.clone(),
            &mut registry,
        );
        let event_listener = fx_event_bus::Listener::new(app.pool.clone(), registry);

        // --- Job listener setup ---
        let jobs_listener = fx_mq_jobs::Listener::new(
            app.pool.clone(),
            fx_durable_ga::register_job_handlers(&app.svc, fx_mq_jobs::RegistryBuilder::new()),
            conf.workers,
            conf.host_id,
            conf.lease_seconds,
        )
        .await?;

        app.tasks = Some(Tasks::start(event_listener, jobs_listener));

        Ok(app)
    }

    pub fn get_svc(&self) -> Arc<fx_durable_ga::optimization::Service> {
        self.svc.clone()
    }

    async fn new(conf: Conf) -> anyhow::Result<Self> {
        tracing_subscriber::fmt()
            .pretty()
            .with_thread_ids(true)
            .with_max_level(tracing::Level::INFO)
            .init();

        let database_url = match &conf {
            Conf::Client(ClientConfig { database_url }) => database_url,
            Conf::Server(ServerConfig { database_url, .. }) => database_url,
        };

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
        let svc = Arc::new(fx_durable_ga::bootstrap(pool.clone()).await?.build());

        Ok(Self {
            pool,
            svc,
            tasks: None,
        })
    }

    pub async fn stop(self) {
        if let Some(tasks) = self.tasks {
            tasks.stop().await;
        }
    }
}
