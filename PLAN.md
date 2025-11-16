# Implementation Plan: Timeseries Training Library

## Overview

This project is building a library to train neural network models on timeseries data. The architecture separates:
- **Generic/Core code**: Reusable across all domains (preprocessing, models, training loops)
- **Domain-specific code**: Data loading, phenotype definitions, evaluators

Each domain will integrate with `fx-durable-ga` for distributed genetic algorithm optimization.

## Current State

### Working Code
- ✅ `src/optimizations/beijing_air_quality/` - Complete, functional training pipeline
  - `commands.rs` - CLI with Train/Export/Infer commands
  - `train.rs` - Generic training loop
  - `model.rs` - Generic models (FeedForward, SimpleRnn, SimpleLstm)
  - `preprocessor.rs` - Generic transforms (ZScore, ROC, Sin, Cos, EMA, Std)
  - `dataset.rs` - Generic DatasetBuilder and SequenceDataset
  - `parser.rs` - Beijing-specific CSV parsing
  - `batcher.rs` - Generic sequence batcher
  - `train_config.rs` - Config serialization
  - `infer_dataset.rs`, `inference.rs` - Inference functionality

- ✅ `src/optimizations/feng.rs` - Old approach (spawns external binary)
- ✅ `src/config.rs` - Service configuration and GA registration
- ✅ `src/bin/client.rs` - Client CLI for GA operations
- ✅ `src/bin/server.rs` - Server that runs GA workers

### Integration Needed
The beijing_air_quality code needs to be integrated with fx-durable-ga so it can be invoked programmatically for optimization.

---

## Step 1: Wire Up Beijing Air Quality (Minimal Changes)

**Goal**: Get beijing_air_quality working with fx-durable-ga without breaking existing code.

### 1.1 Create Phenotype

**File**: `src/optimizations/beijing_air_quality/phenotype.rs`

**Purpose**: Define the variable parameters that will be optimized by the genetic algorithm.

**Implementation Strategy**: Mirror `feng::Config` exactly to start.

**Optimizable Parameters** (31 genes total):
- **7 features** (28 genes): Each feature has:
  - Source column index (0-14) from VALID_COLUMNS (15 columns total)
  - Pipeline length (0-2)
  - Transform 1 (0-11): ZScore, ROC, Std with various windows
  - Transform 2 (0-11): Same as above
- **3 hyperparameters** (3 genes):
  - Hidden size: [4, 8, 16, 32, 64, 128]
  - Learning rate: [1e-4, 5e-4, 1e-3]
  - Sequence length: [10, 20, 30, ..., 100] (step 10)

**Fixed Parameters** (not optimized):
- Target: Always TEMP (user specifies target)
- Model architecture: FeedForward (hardcoded for now)
- Batch size: 100
- Epochs: 25
- Prediction horizon: 1
- Time features: Always include hour_sin, hour_cos, month_sin, month_cos

**Contents**:
- Struct with same structure as `feng::Config`:
  ```rust
  pub struct BeijingPhenotype {
      #[serde(skip)]
      features: Vec<Feature>,  // The optimized features with pipelines
      pub hidden_size: usize,
      pub learning_rate: f64,
      pub sequence_length: usize,
  }
  ```
- Internal `Transform` enum (12 hardcoded variants mapping genes to specific transforms):
  - ZScore10, ZScore24, ZScore48, ZScore96
  - Roc1, Roc4, Roc8, Roc12
  - Std10, Std24, Std48, Std96
- Internal `Feature` struct:
  ```rust
  struct Feature {
      source: String,        // e.g., "TEMP", "PRES"
      transforms: Vec<Transform>,  // Pipeline of transforms to apply
  }
  ```
- **Transform methods**:
  - `from_gene(i64) -> Option<Transform>` - decode gene to variant
  - `to_gene(self) -> i64` - encode variant to gene
  - `to_node(&self) -> preprocessor::Node` - **NEW**: convert to beijing preprocessor node
    ```rust
    fn to_node(&self) -> super::preprocessor::Node {
        match self {
            Self::ZScore10 => Node::ZScore(ZScore::new(10)),
            Self::Roc1 => Node::Roc(Roc::new(1)),
            Self::Std24 => Node::Std(Std::new(24)),
            // ... etc for all 12 variants
        }
    }
    ```
- **Feature methods**:
  - `to_pipeline(&self) -> preprocessor::Pipeline` - convert transforms to Pipeline:
    ```rust
    fn to_pipeline(&self) -> Pipeline {
        let nodes = self.transforms.iter()
            .map(|t| t.to_node())
            .collect();
        Pipeline::new(nodes)
    }
    ```
- Implement `fx_durable_ga::models::Encodeable` trait
  - `NAME`: "beijing_air_quality_feature_engineering"
  - `morphology()`: Define gene bounds
    ```rust
    fn morphology() -> Vec<GeneBounds> {
        let mut bounds = Vec::new();
        // 7 features, each with 4 genes
        for _ in 0..7 {
            bounds.push(GeneBounds::integer(0, 14, 15).unwrap()); // source (15 columns)
            bounds.push(GeneBounds::integer(0, 2, 3).unwrap());   // pipeline_length
            bounds.push(GeneBounds::integer(0, 11, 12).unwrap()); // transform_1
            bounds.push(GeneBounds::integer(0, 11, 12).unwrap()); // transform_2
        }
        // Hyperparameters
        bounds.push(GeneBounds::integer(0, 5, 6).unwrap());   // hidden_size
        bounds.push(GeneBounds::integer(0, 2, 3).unwrap());   // learning_rate
        bounds.push(GeneBounds::integer(0, 9, 10).unwrap());  // sequence_length
        bounds
    }
    ```
  - `encode()`: Phenotype → genes
  - `decode()`: genes → Phenotype
- Import `SOURCE_COLUMNS` from ingestion module: `use super::ingestion::SOURCE_COLUMNS;`

### 1.2 Create Evaluator

**File**: `src/optimizations/beijing_air_quality/evaluator.rs`

**Purpose**: Evaluate a phenotype by training a model and returning fitness score.

**Key difference from feng**: Instead of spawning external binary, call training code **directly** within the same process.

**Implementation**:
- Unit struct implementing `fx_durable_ga::models::Evaluator<Phenotype>`
- `fitness()` method signature (async, returns BoxFuture):
  ```rust
  fn fitness<'a>(
      &self,
      genotype_id: Uuid,
      phenotype: BeijingPhenotype,
      _terminated: &'a Box<dyn Terminated>,
  ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>>
  ```

**Implementation steps**:
1. Create Backend and device inside fitness:
   ```rust
   type Backend = Autodiff<NdArray>;
   let device = NdArrayDevice::default();
   ```

2. Convert phenotype to feature/target definitions:
   - Build time features (always included):
     ```rust
     let time_features = vec![
         ("hour_sin".to_string(), "hour".to_string(), Pipeline::new(vec![Node::Sin(Sin::new(24.0))])),
         ("hour_cos".to_string(), "hour".to_string(), Pipeline::new(vec![Node::Cos(Cos::new(24.0))])),
         ("month_sin".to_string(), "month".to_string(), Pipeline::new(vec![Node::Sin(Sin::new(12.0))])),
         ("month_cos".to_string(), "month".to_string(), Pipeline::new(vec![Node::Cos(Cos::new(12.0))])),
     ];
     ```
   - Convert phenotype features to feature definitions:
     ```rust
     let mut features = time_features;
     for (i, feat) in phenotype.features.iter().enumerate() {
         let name = format!("feat_{}", i);
         let pipeline = feat.to_pipeline();  // Uses Transform::to_node()
         features.push((name, feat.source.clone(), pipeline));
     }
     ```
   - Target definition:
     ```rust
     let targets = vec![(
         "target_temp".to_string(),
         "TEMP".to_string(),
         Pipeline::new(vec![])  // No preprocessing on target
     )];
     ```

3. Load data from all stations using `super::ingestion::PATHS` constant:
   - For each path, call `dataset::build_dataset_from_file(path, &features, &targets)`
   - Split 80/20 train/validation using `.build(sequence_length, 1, Some(0.8))`
     - `sequence_length` from phenotype
     - `prediction_horizon = 1` (hardcoded - predict 1 timestep ahead)
     - `0.8` split ratio (80% train, 20% validation)
   - Combine all stations into unified datasets using `SequenceDataset::from_items()`

4. Create FeedForward model:
   - input_size = 11 (4 time features + 7 optimized features)
   - hidden_size from phenotype
   - output_size = 1 (single target)
   - sequence_length from phenotype
   - Note: Each feature pipeline outputs a single f32 value

5. Train using `train::train()`:
   - batch_size: 100 (fixed)
   - epochs: 25 (fixed)
   - learning_rate from phenotype
   - model_save_path: None (don't save during GA optimization)
   - train_config: None (don't save config during GA)

6. Return validation loss:
   ```rust
   // train() returns (model, f32)
   let (_model, best_valid_loss) = train::train(...);
   Ok(best_valid_loss as f64)  // TODO: Cast f32 to f64 for database - align types later
   ```

**Notes**:
- Lower validation loss = better fitness (minimize)
- Use `Box::pin(async move { ... })` to wrap implementation
- Import PATHS from `super::ingestion::PATHS`
- The key is converting phenotype's Transform enum → preprocessor::Node → Pipeline

### 1.3 Create Ingestion Module

**File**: `src/optimizations/beijing_air_quality/ingestion.rs`

**Purpose**: Centralize domain-specific data ingestion constants and logic for Beijing air quality domain.

**Contents**:
```rust
// Valid source columns from Beijing air quality CSV files
// These correspond to the CSV header columns (excluding No, year, station which are metadata)
// CSV header: No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station
pub const SOURCE_COLUMNS: &[&str] = &[
    "day", "hour", "month", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
    "RAIN", "wd", "WSPM",
];

// CSV file paths for Beijing air quality data
// These are relative to the project root
pub const PATHS: &[&str] = &[
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Changping_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Dingling_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Dongsi_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Guanyuan_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Gucheng_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Huairou_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Shunyi_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Tiantan_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Wanliu_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Wanshouxigong_20130301-20170228.csv",
];

pub const WANSHOUXIGONG_PATH: &str = "src/optimizations/beijing_air_quality/data/PRSA_Data_Wanshouxigong_20130301-20170228.csv";
```

**Note**: This module contains domain-specific ingestion logic for Beijing air quality. Other domains will have their own ingestion modules with their own SOURCE_COLUMNS and file paths.

**Future**: CSV parsing logic from parser.rs could eventually move here.

### 1.4 Update Module Exports

**File**: `src/optimizations/beijing_air_quality/mod.rs`

Add:
```rust
pub mod ingestion;
pub mod phenotype;
pub mod evaluator;

pub use phenotype::BeijingPhenotype;
pub use evaluator::BeijingEvaluator;
```

### 1.5 Register with GA Service

**File**: `src/config.rs`

In the `App::new()` method, register the Beijing evaluator:

```rust
let svc = Arc::new(
    fx_durable_ga::bootstrap(pool.clone())
        .await?
        .register::<feng::Config, _>(FengEvaluator)
        .register::<beijing_air_quality::BeijingPhenotype, _>(beijing_air_quality::BeijingEvaluator)
        .await?
        .build(),
);
```

### 1.6 Restructure Commands Module

**File**: `src/optimizations/beijing_air_quality/commands.rs`

**Goal**: Simplify to focus on "RequestOptimization" command for this domain.

**Changes**:

1. **Simplify the enum** - keep only optimization request:
   ```rust
   #[derive(Debug, Subcommand)]
   pub enum BeijingCommand {
       /// Request a genetic algorithm optimization for Beijing air quality prediction
       RequestOptimization {
           // Note: type_name is NOT a parameter - it's hardcoded to BeijingPhenotype::NAME
           
           /// Fitness goal: MIN(threshold) or MAX(threshold)
           #[arg(long, required = true, value_parser = parse_fitness_goal)]
           fitness_goal: FitnessGoal,
           
           /// Schedule: GENERATIONAL(generations, population) or ROLLING(evaluations, population, interval)
           #[arg(long, required = true, value_parser = parse_schedule)]
           schedule: Schedule,
           
           /// Selector: TOURNAMENT(tournament_size, sample_size) or ROULETTE(sample_size)
           #[arg(long, required = true, value_parser = parse_selector)]
           selector: Selector,
           
           /// Mutagen: MUTAGEN(temperature, mutation_rate)
           #[arg(long, required = true, value_parser = parse_mutagen)]
           mutagen: Mutagen,
           
           /// Initial population size using latin hypercube
           #[arg(long, required = true)]
           initial_population: u32,
       }
   }
   ```

2. **Remove Train/Export/Infer commands** for now:
   - These can be added back later in a more fitting shape
   - Focus: Starting a GA optimization is the main goal
   - Train command logic is now in the evaluator

3. **Remove `Args` struct and `Parser`** - only export `BeijingCommand` subcommand

4. **Remove `main()` function** - no longer a binary entry point

5. **Add async execute method**:
   ```rust
   impl BeijingCommand {
       pub async fn execute(self, svc: Arc<fx_durable_ga::optimization::Service>) -> anyhow::Result<()> {
           match self {
               Self::RequestOptimization {
                   fitness_goal,
                   schedule,
                   selector,
                   mutagen,
                   initial_population,
               } => {
                   // Hardcode type_name from BeijingPhenotype::NAME
                   let type_name = BeijingPhenotype::NAME;
                   let type_hash: i32 = fnv1a_hash_str_32(type_name) as i32;
                   
                   svc.new_optimization_request(
                       type_name,
                       type_hash,
                       fitness_goal,
                       schedule,
                       selector,
                       mutagen,
                       Crossover::single_point(),
                       Distribution::latin_hypercube(initial_population),
                   ).await?;
                   
                   tracing::info!("Started optimization request for Beijing air quality");
                   Ok(())
               }
           }
       }
   }
   ```

6. **Keep and update helper code**:
   - Move `PATHS` to `ingestion.rs` (used by evaluator)
   - **Keep** value parsers needed for RequestOptimization:
     - `parse_fitness_goal()` (copy from client.rs)
     - `parse_schedule()` (copy from client.rs)
     - `parse_selector()` (copy from client.rs)
     - `parse_mutagen()` (copy from client.rs)
   - **Add required imports** for parsers and execute:
     ```rust
     use fx_durable_ga::models::{FitnessGoal, Schedule, Selector, Mutagen, Crossover, Distribution};
     use const_fnv1a_hash::fnv1a_hash_str_32;
     use std::sync::Arc;
     use super::phenotype::BeijingPhenotype;
     ```
   - Remove `parse_feature_pipeline()` (not needed anymore)
   - Remove `ResultOutput` (was for CLI training output)
   - Remove `Args`, `Parser` (not a standalone binary)
   - Remove `main()` function
   - Remove `PATHS`, `WANSHOUXIGONG_PATH` (moved to ingestion.rs)

**Note**: The RequestOptimization command needs access to the GA service, so we pass it in. The type_name is hardcoded from `BeijingPhenotype::NAME`.

### 1.7 Integrate Beijing Commands into Client CLI

**File**: `src/bin/client.rs`

**Goal**: Add Beijing domain as a subcommand, keep existing commands unchanged.

**Changes**:

1. **Add import**:
   ```rust
   use fx_durable_ga_app::optimizations::beijing_air_quality;
   ```

2. **Add Beijing to Command enum**:
   ```rust
   #[derive(Debug, Subcommand)]
   enum Command {
       // Existing GA operations (keep unchanged)
       RequestOptimization { /* ... */ },
       ListGenotypes { /* ... */ },
       GetPhenotype { /* ... */ },
       
       // Beijing domain-specific operations
       #[command(subcommand)]
       Beijing(beijing_air_quality::BeijingCommand),
   }
   ```

3. **Handle Beijing commands in main()**:
   ```rust
   match args.command {
       Command::Beijing(cmd) => {
           cmd.execute(client.get_svc()).await?;
       }
       Command::RequestOptimization { /* ... */ } => {
           // existing logic (unchanged)
       }
       // ... other commands (unchanged)
   }
   ```

**Notes**:
- Existing commands remain unchanged and should continue to work
- Beijing command needs the service, so we pass `client.get_svc()`
- No changes needed to GetPhenotype for now - can be enhanced later

### 1.8 Verification

Test that:
1. Server starts without errors
2. Can request optimization for Beijing domain via client
3. Workers pick up jobs and train models
4. Fitness scores are recorded
5. Can list genotypes and retrieve phenotypes

**Success criteria**: Beijing domain is fully integrated with GA without breaking any existing functionality.

---

## Step 2: Extract Core Components (Light Refactoring)

**Goal**: Improve code organization by separating generic from domain-specific code.

### 2.1 Create Core Module Structure

**New directory**: `src/core/`

Move generic code (no logic changes, just reorganization):
- `preprocessor.rs` - Generic transforms (from beijing_air_quality)
- `model.rs` - Model trait and implementations
- `train.rs` - Generic training loop
- `batcher.rs` - Generic sequence batcher
- `dataset.rs` - Generic dataset builder (may need minor refactoring)
- `types.rs` - Shared type aliases:
  ```rust
  pub type Feature = Vec<f32>;
  pub type Timestep = Vec<Feature>;
  pub type Sequence = Vec<Timestep>;
  ```

**File**: `src/core/mod.rs`
```rust
pub mod batcher;
pub mod dataset;
pub mod model;
pub mod preprocessor;
pub mod train;
pub mod types;
```

### 2.2 Update Beijing Imports

Update imports in `src/optimizations/beijing_air_quality/`:
```rust
use crate::core::{model, train, dataset, preprocessor, batcher};
```

### 2.3 Rename for Clarity (Optional)

Consider renaming files in beijing_air_quality:
- `parser.rs` → `data_source.rs` (more descriptive)

### 2.4 Verification

- All tests still pass
- No functional changes
- Code is better organized

---

## Step 3: Prepare for Additional Domains

**Goal**: Validate that the architecture supports multiple domains easily.

### 3.1 Document Domain Interface

Each domain module must provide:

1. **Phenotype** - Struct implementing `Encodeable`
   - Defines optimizable parameters
   - Gene encoding/decoding logic

2. **Evaluator** - Struct implementing `Evaluator<Phenotype>`
   - Takes phenotype, returns fitness score
   - Uses core components for training

3. **Registration function** (optional pattern):
   ```rust
   pub fn register(builder: ServiceBuilder) -> ServiceBuilder {
       builder.register::<MyPhenotype, _>(MyEvaluator)
   }
   ```

4. **Commands enum** - Derives `clap::Subcommand`
   - Domain-specific CLI commands
   - Train, export, infer, etc.

### 3.2 Update Config Pattern

Make registration more modular:
```rust
let mut builder = fx_durable_ga::bootstrap(pool.clone()).await?;
builder = beijing_air_quality::register(builder).await?;
builder = other_domain::register(builder).await?;
let svc = Arc::new(builder.build());
```

---

## Future Considerations

### Data Format Expectations

All domains are expected to work with 3D tensors:
```
[
  [  // Timestep 1
    [f32, f32, ..], // Feature 1
    [f32, f32, ..], // Feature 2
  ],
  [  // Timestep 2
    [f32, f32, ..], // Feature 1
    [f32, f32, ..], // Feature 2
  ],
]
```

### Preprocessing Pipeline

**Generic part** (in core):
- Transform nodes: ZScore, ROC, Sin, Cos, EMA, Std
- Pipeline struct for chaining transforms
- Processing HashMap<String, f32> through pipeline

**Domain-specific part**:
- Data loading from source (CSV, API, database, etc.)
- Parsing raw data into HashMap<String, f32>
- Valid column definitions
- Feature engineering specific to domain

### Clean Separation

```
Domain Data Source
      ↓
HashMap<String, f32>  ← Domain boundary
      ↓
Core Preprocessing Pipeline
      ↓
Core Dataset Builder
      ↓
Core Training
      ↓
Fitness Score
      ↓
Domain Evaluator returns to GA
```

---

## Notes

- **Principle**: Preserve working code, refactor incrementally
- **Testing**: Verify after each step before proceeding
- **Feng module**: Can be removed after Beijing is working (old approach)
- **Commands.rs**: Current main() function logic moves into evaluator
- **Model choice**: Each domain can choose FeedForward/RNN/LSTM
- **Directory naming**: Could rename `src/optimizations/` to `src/domains/` for clarity

---

## Questions to Resolve

1. Should preprocessing transforms support multi-dimensional features, or stay with Vec<f32>?
2. How to handle domain-specific data sources? Trait-based or just convention?
3. Should we create a `Domain` trait to formalize the interface?
4. Better name for `src/optimizations/` directory?
