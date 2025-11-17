# fx-durable-ga-app

Distributed genetic algorithm optimization for timeseries neural network training.

## Overview

This application uses a distributed genetic algorithm to optimize neural network hyperparameters and feature engineering for timeseries prediction tasks. Currently supports:

- **Beijing Air Quality**: Temperature prediction using air quality sensor data from 12 stations
- **Feng (legacy)**: External binary-based optimization

## Architecture

```
Client CLI → PostgreSQL Queue → Worker Pool → Training → Fitness Scores
```

- **Client**: Submit optimization requests and query results
- **Server**: Worker pool that picks up jobs and evaluates phenotypes
- **Database**: PostgreSQL stores genotypes, phenotypes, and fitness scores

## Setup

### Prerequisites
- Rust (latest stable)
- PostgreSQL database
- Environment files configured

### Environment Configuration

Create `.env.client` in `src/bin/`:
```env
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ga
```

Create `.env.server` in `src/bin/`:
```env
DATABASE_URL=postgres://postgres:postgres@localhost:5432/ga
HOST_ID=00000000-0000-0000-0000-000000000001
LEASE_SECONDS=300
```

### Build

```bash
SQLX_OFFLINE=true cargo build --release
```

## Usage

### 1. Start the Server

```bash
cargo run --release --bin server
```

The server will start workers that process optimization jobs.

### 2. Submit an Optimization Request

#### Beijing Air Quality Optimization

**Quick Test (Small Scale):**
```bash
cargo run --release --bin client -- beijing request-optimization \
  --fitness-goal 'MIN(15.0)' \
  --schedule 'GENERATIONAL(5,20)' \
  --selector 'TOURNAMENT(3,5)' \
  --mutagen 'MUTAGEN(1.0,0.15)' \
  --initial-population 20
```

**Standard Run (Medium Scale):**
```bash
cargo run --bin client -- beijing request-optimization \
  --fitness-goal 'MIN(10.0)' \
  --schedule 'GENERATIONAL(20,50)' \
  --selector 'TOURNAMENT(5,10)' \
  --mutagen 'MUTAGEN(1.0,0.1)' \
  --initial-population 100
```

**Thorough Search (Large Scale):**
```bash
cargo run --bin client -- beijing request-optimization \
  --fitness-goal 'MIN(5.0)' \
  --schedule 'GENERATIONAL(50,100)' \
  --selector 'TOURNAMENT(7,20)' \
  --mutagen 'MUTAGEN(0.8,0.05)' \
  --initial-population 200
```

**Rolling Window Approach:**
```bash
cargo run --bin client -- beijing request-optimization \
  --fitness-goal 'MIN(8.0)' \
  --schedule 'ROLLING(500,40,20)' \
  --selector 'ROULETTE(15)' \
  --mutagen 'MUTAGEN(1.2,0.12)' \
  --initial-population 80
```

#### Generic Request (Any Registered Type)

```bash
cargo run --bin client -- request-optimization \
  --type-name feature_engineering \
  --fitness-goal 'MIN(0.0)' \
  --schedule 'GENERATIONAL(10,10)' \
  --selector 'TOURNAMENT(5,20)' \
  --mutagen 'MUTAGEN(0.6,0.3)' \
  --initial-population 20
```

### 3. Query Results

**List all genotypes for a request:**
```bash
cargo run --bin client -- list-genotypes --request-id <REQUEST_UUID>
```

Example output:
```
Genotypes for request 12345678-1234-1234-1234-123456789abc:

Genotype ID                          Fitness
-------------------------------------------------------
abcdef01-2345-6789-abcd-ef0123456789 8.234567
fedcba98-7654-3210-fedc-ba9876543210 9.876543
...
```

**Get detailed phenotype information:**
```bash
cargo run --bin client -- get-phenotype --genotype-id <GENOTYPE_UUID>
```

Example output:
```json
{
  "hidden_size": 64,
  "learning_rate": 0.0005,
  "sequence_length": 50
}
```

## Parameter Guide

### Fitness Goal
- `MIN(threshold)`: Minimize metric (e.g., loss)
- `MAX(threshold)`: Maximize metric (e.g., accuracy)

### Schedule
- `GENERATIONAL(generations, population)`: Classic generational GA
  - `generations`: Number of generations to evolve
  - `population`: Number of individuals per generation
- `ROLLING(evaluations, population, interval)`: Rolling window approach
  - `evaluations`: Total number of evaluations
  - `population`: Population size
  - `interval`: Report progress every N evaluations

### Selector
- `TOURNAMENT(tournament_size, sample_size)`: Tournament selection
  - `tournament_size`: Number of individuals per tournament
  - `sample_size`: Number of parents to select
- `ROULETTE(sample_size)`: Fitness-proportionate selection
  - `sample_size`: Number of parents to select

### Mutagen
- `MUTAGEN(temperature, mutation_rate)`: Mutation parameters
  - `temperature`: Mutation strength (higher = more exploration)
  - `mutation_rate`: Probability of mutation per gene (0.0-1.0)

### Initial Population
- Number of individuals to generate using Latin Hypercube sampling
- Larger values = better initial diversity, but more computation

## Beijing Air Quality Domain

### What Gets Optimized

The genetic algorithm optimizes 31 genes:

**7 Features (28 genes):** Each feature has:
- Source column (15 options): month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM
- Pipeline length (0-2 transforms)
- Transform 1 (12 options): ZScore, ROC, Std with various windows
- Transform 2 (12 options): Same as above

**3 Hyperparameters (3 genes):**
- Hidden size: [4, 8, 16, 32, 64, 128]
- Learning rate: [1e-4, 5e-4, 1e-3]
- Sequence length: [10, 20, 30, ..., 100]

### Fixed Parameters
- **Target**: Temperature (TEMP)
- **Model**: FeedForward neural network
- **Time features**: hour_sin, hour_cos, month_sin, month_cos (always included)
- **Batch size**: 100
- **Epochs**: 25
- **Data**: Combined from 12 Beijing air quality monitoring stations
- **Split**: 80% training, 20% validation

### Training Process

1. Load data from 12 weather stations
2. Apply feature engineering pipelines
3. Split into train/validation sets
4. Train FeedForward model
5. Return validation MSE loss as fitness

Lower validation loss = better fitness.

## Development

### Project Structure

```
src/
├── bin/
│   ├── client.rs          # CLI client
│   └── server.rs          # Worker server
├── config.rs              # App configuration & GA registration
├── optimizations/
│   ├── feng.rs            # Legacy optimization (external binary)
│   └── beijing_air_quality/
│       ├── phenotype.rs   # Encodeable phenotype (31 genes)
│       ├── evaluator.rs   # Evaluator (trains model, returns fitness)
│       ├── ingestion.rs   # Data paths and column definitions
│       ├── commands.rs    # CLI subcommands
│       ├── model.rs       # Neural network models
│       ├── train.rs       # Training loop
│       ├── dataset.rs     # Dataset builder
│       ├── preprocessor.rs # Transform nodes
│       └── ...
└── lib.rs
```

### Adding a New Domain

To add a new optimization domain:

1. Create module in `src/optimizations/your_domain/`
2. Implement `Encodeable` trait for your phenotype
3. Implement `Evaluator<YourPhenotype>` trait
4. Create `commands.rs` with CLI subcommand
5. Register in `src/config.rs`
6. Add subcommand to `src/bin/client.rs`

See `beijing_air_quality/` as reference implementation.

## License

MIT
