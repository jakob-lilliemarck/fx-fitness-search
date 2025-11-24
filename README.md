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
SQLX_OFFLINE=true cargo build --release --all
```

### Migrate

```bash
target/release/migrate
```

## Usage

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
  --prediction-horizon 1
```
