# fx-durable-ga-app

Distributed genetic algorithm optimization for time series model training, with optional embedding-based indexing of genotypes.

## Prerequisites
- PostgreSQL
- Environment variables (see below)

## Environment Configuration

Example `client` variables:

```
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ga?options=-c%20search_path%3Dfx_mq_jobs%2Cfx_event_bus%2Cfx_durable_ga
MODEL_SAVE_PATH=/absolute/path/for/models
```

Example `server` variables:

```
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ga?options=-c%20search_path%3Dfx_mq_jobs%2Cfx_event_bus%2Cfx_durable_ga
MODEL_SAVE_PATH=/absolute/path/for/models
HOST_ID=00000000-0000-0000-0000-000000000001
LEASE_SECONDS=450
SHUTDOWN_TIMEOUT_SECONDS=5
MAX_WORKERS=4
BATCH_SIZE=128
```

## Build

Default (WGPU backend, GPU-accelerated with CPU fallback):

```
cargo build --release --all
```

CPU-only (ndarray backend):

```
cargo build --release --no-default-features --features backend-ndarray
```

## Run

Start server:

```
target/release/server
```

Submit a small optimization request (current CLI syntax):

```
target/release/client beijing request-optimization \
  --fitness-goal 'MIN(0.0)' \
  --schedule 'GENERATIONAL(20, 8)' \
  --selector 'TOURNAMENT(2)' \
  --mutation-rate 0.1 \
  --temperature 0.25 \
  --prediction-horizon 1 \
  --epochs 8 \
  --patience 3 \
  --validation-start-epoch 3
```

## Embedding Sanity Checks (High-Level)

When indexing is enabled, each evaluated genotype should have an embedding. You can sanity-check quality with three simple checks:

1) Coverage
- For a recent request, verify that all evaluated genotypes have corresponding embeddings.

2) Structure vs random
- Compare nearest-neighbor distances to a random baseline. Nearest neighbors should be significantly closer than random pairs.

3) Semantic consistency
- Compare genome JSON for nearest vs farthest neighbors. Nearest neighbors should share similar feature pipelines; farthest should be structurally different.

These checks do not require code changes and can be done directly in the database using the embedding tags (e.g., `genotype_id:*`).

## Data Source

- Beijing Multi-Site Air Quality: Chen, S. (2017). UCI ML Repository.
  DOI: 10.24432/C5RK5G (CC BY 4.0)
