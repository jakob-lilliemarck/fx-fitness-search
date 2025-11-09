use futures::future::join_all;
use fx_durable_ga_app::config::{ServerConfig, bootstrap};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let conf = ServerConfig::from_env()?;

    let (service, pool) = bootstrap(&conf.database_url).await?;

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
        conf.workers,
        conf.host_id,
        conf.lease_seconds,
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
