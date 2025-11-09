use futures::future::join_all;
use sqlx::types::Uuid;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    // Get the database url
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    // Get and parse host id
    // The host id uniquely identifies the machine that handled a job or event
    let host_id_str = std::env::var("HOST_ID").expect("HOST_ID must be set");
    let host_id = Uuid::parse_str(&host_id_str).expect("HOST_ID must be a valid UUID");

    // Get and parse the lease duration in seconds
    // This variable determines how long a worker has exclusive access to its work
    // Workers can renew it, but if they don't, the lease should be longer than time the task is expected to take
    let lease_seconds_str = std::env::var("LEASE_SECONDS")?;
    let lease_seconds = lease_seconds_str
        .parse::<u64>()
        .expect("LEASE_SECONDS must be a valud u64");
    let lease_duration = Duration::from_secs(lease_seconds);

    // Get the number of physical cores on the current machine.
    // At the time of writing this I know that my current training with NdArray runs on 2 cores.
    // As such we divide by two and round down, to get the numbers of worker to run.
    // Note that this couples the logic of this server to knowledge of the training,
    // which obviously is less than ideal - however, this is just an experiment.
    let physical_cores = num_cpus::get_physical();
    let workers = physical_cores / 2;

    // Create a database connection pool
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    // Run required migrations migrations
    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_jobs::run_migrations(&pool, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME).await?;
    fx_durable_ga::run_migrations(&pool).await?;

    // Create the GA service and wrap it in an Arc
    let service = std::sync::Arc::new(fx_durable_ga::bootstrap(pool.clone()).await?.build());

    // --- Event listener setup ---
    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    fx_durable_ga::register_event_handlers(
        std::sync::Arc::new(fx_mq_jobs::Queries::new(fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME)),
        service.clone(),
        &mut registry,
    );
    let mut event_listener = fx_event_bus::Listener::new(pool.clone(), registry);

    // --- Job listener setup ---
    let mut jobs_listener = fx_mq_jobs::Listener::new(
        pool.clone(),
        fx_durable_ga::register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        workers,
        host_id,
        lease_duration,
    )
    .await?;

    // --- Cancellation token ---
    let shutdown_token = tokio_util::sync::CancellationToken::new();
    let shutdown_token_jobs = shutdown_token.clone();
    let shutdown_token_events = shutdown_token.clone();

    // --- Spawn event listener task ---
    let event_handle = tokio::spawn(async move {
        tokio::select! {
            res = event_listener.listen(None) => res.map_err(|e| anyhow::anyhow!(e)),
            _ = shutdown_token_events.cancelled() => {
                tracing::info!("Event listener received shutdown signal");
                Ok(())
            }
        }
    });

    // --- Spawn job listener task ---
    let job_handle = tokio::spawn(async move {
        tokio::select! {
            res = jobs_listener.listen() => res.map_err(|e| anyhow::anyhow!(e)),
            _ = shutdown_token_jobs.cancelled() => {
                tracing::info!("Job listener received shutdown signal");
                jobs_listener.stop(Duration::from_secs(10)).await.map_err(|e| anyhow::anyhow!(e))
            }
        }
    });

    // --- Wait for Ctrl+C ---
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutdown requested, cancelling listeners...");

    shutdown_token.cancel(); // signal all listeners

    // --- Wait for tasks to finish ---
    let results: Vec<anyhow::Result<()>> = join_all(vec![job_handle, event_handle])
        .await
        .into_iter()
        .map(|res| res.unwrap_or_else(|e| Err(anyhow::anyhow!(e)))) // handle join errors
        .collect();

    for result in results {
        if let Err(err) = result {
            tracing::error!("Listener finished with error: {:?}", err);
        }
    }

    tracing::info!("Shutdown complete");
    Ok(())
}
