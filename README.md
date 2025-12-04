# fx-durable-ga-app

Distributed genetic algorithm optimization for timeseries neural network training.

## Architecture

- **Client**: Submit optimization requests and query results
- **Server**: Worker pool that picks up jobs and evaluates phenotypes
- **Database**: PostgreSQL stores genotypes, phenotypes, and fitness scores

## Setup
Clone the repo and install or compile it yourself, or download one of the precompiled release binaries from

https://github.com/jakob-lilliemarck/fx-fitness-search/releases

### Prerequisites
- PostgreSQL database
- Environment variable configurations

### Environment Configuration
#### Example `client` variables:
```
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ga?options=-c%20search_path%3Dfx_mq_jobs%2Cfx_event_bus%2Cfx_durable_ga
MODEL_SAVE_PATH=/some/absolute/path
```

#### Example `server` variables:
```
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ga?options=-c%20search_path%3Dfx_mq_jobs%2Cfx_event_bus%2Cfx_durable_ga
MODEL_SAVE_PATH=/some/absolute/path
HOST_ID=00000000-0000-0000-0000-000000000001
LEASE_SECONDS=450
SHUTDOWN_TIMEOUT_SECONDS=5
MAX_WORKERS=4
```

### Migrate
The migrate binary ensures all migrations are run on the database. You can run it manually, or it will run on server startup or at the first CLI command.
```bash
target/release/migrate
```

## Usage
Usage examples assume the binaries were compiled locally using `cargo build --release --all` and run from the directory root.

### 1. Start the Server

```bash
target/release/server
```

The server will start workers that process optimization jobs.

### 2. Submit an Optimization Request

#### Beijing Air Quality Optimization

```bash
target/release/client beijing request-optimization \
  --fitness-goal 'MIN(0.0)' \
  --schedule 'GENERATIONAL(100, 12)' \
  --selector 'TOURNAMENT(3, 12)' \
  --mutagen 'MUTAGEN(0.7, 0.4)' \
  --initial-population 50 \
  --prediction-horizon 1 \
  --epochs 20 \
  --patience 3 \
  --validation-start-epoch 5 \
  --batch-size 128
```

### Data Sources

- **Beijing Multi-Site Air Quality**: Chen, S. (2017). [Dataset]. UCI Machine Learning Repository.
  DOI: [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G) |
  License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
