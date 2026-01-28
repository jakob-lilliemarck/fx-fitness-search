# Introduce GP-based preprocessing pipeline
Beijing optimization currently encodes every feature as an `Extract` pipeline (interpolation + ordered `Node`s) inside `BeijingPhenotype` (`src/optimizations/beijing_air_quality/evaluator.rs`). This array-based genome causes invariants to break when gene blocks are mutated mid-block. We need to represent feature selection and preprocessing as an actual genetic program that safely composes stateful preprocessing nodes, supports dynamic output counts, and plugs into ingestion/train config.

## Current state
`Extract` (`src/core/ingestion/extract.rs`) holds `source`, `Interpolation`, and `Vec<Node>`. `file_csv::Csv::process_row` executes interpolation + nodes sequentially for each feature/target column, with `Interpolation::Error` failing on missing data.
`Node` (`src/core/preprocessing/node.rs`) is a thin wrapper around stateful processors (EMA, ZScore, etc.) plus Noop.
`BeijingPhenotype` maintains `Vec<Extract>` features + hyperparameters; GA logic mutates/crosses over by editing those vectors.
Reference repo (`reference/fx-genetic-programming/src/lib.rs`) shows a stateless GP with `Select` leaves and multi-output `Program`.

## Proposed changes
### 1. Core GP data structures
Create `src/core/preprocessing/gp.rs` exporting:
* `GpError` (thiserror) for missing keys/interpolation/processing errors.
* `GpNode` enum: `Select { key: String }`, `Interpolate(Interpolation)`, `Compute(Node)`, `Chain(Vec<GpNode>)`, `Output(Vec<GpNode>)` (or similar) enabling tree graphs where leaves are `Select` and chains model ordered preprocessing.
* Each node owns its state (Interpolation and `Node` variants already handle serde skip for state). Implement `process(&mut self, inputs: &HashMap<String, Option<f32>>) -> Result<Option<f32>, GpError>` with enforced `Select -> Interpolate -> Compute*` flow (error if `Select` missing key or `Interpolation` fails).
* `GpProgram { outputs: Vec<GpNode> }` with `evaluate(&mut self, inputs: &HashMap<String, Option<f32>>) -> Result<Vec<f32>, GpError>` ensuring deterministic output order and rejecting `None` leaves.
* Serde impls ensure runtime state isn’t serialized (reuse `#[serde(skip)]`).
Add unit tests covering: select success/missing key failure, interpolation error propagation, compute chaining with EMA/Std, multi-output vector sizing.

### 2. Wire GP program into ingestion
Extend `Csv` (`src/core/ingestion/file_csv.rs`) to optionally hold a `GpProgram` for features (targets remain `Extract` for now). Add constructor overload or new type `ProgrammedCsv`? Simplest: add `Option<GpProgram>` on `Csv` plus `HashMap<String, Option<f32>>` row map forwarded to `GpProgram::evaluate`. Replace feature `process_row` path to:
* Build `HashMap<String, Option<f32>>` from cast row (present values -> Some, missing -> None).
* If program present, call it to get `Vec<f32>`; else fall back to legacy `Extract` flow.
Targets keep using legacy `Extract` until we decide otherwise.
Propagate `GpError` via `ingestion::Error::ProcessingError`.

### 3. Update preprocessing module exports
Register new module in `src/core/preprocessing/mod.rs` to expose `GpNode`, `GpProgram`. Ensure interpolation + node modules are re-used (no duplication).

### 4. Refactor Beijing genotype representation
Modify `BeijingPhenotype` to:
```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeijingPhenotype {
    processing: GpProgram,
    hidden_size: usize,
    learning_rate: f64,
    sequence_length: usize,
}
```
Adjust `random`, `mutate`, `crossover` to manipulate the GP tree instead of `Vec<Extract>` (initially: wrap each randomly chosen feature source into `Select -> interpolation -> compute` chain; future: use richer GP ops). Targets stay as before but feature generation uses `GpProgram` when building `TrainConfig`.

### 5. TrainConfig / evaluation integration
In `evaluate` impl (`src/optimizations/beijing_air_quality/evaluator.rs`):
* When preparing training data, pass the phenotype’s `GpProgram` to ingestion so that feature sequences come from the program outputs.
* Keep baseline positional/time features by composing them inside the GP tree or by appending fixed columns after `GpProgram::evaluate`.
* Ensure serialization/deserialization of phenotype works through GA service (update `decode_genotype`).
Add tests around phenotype randomization + ingestion to ensure `GpProgram` runs end-to-end on sample CSV rows.

### 6. Error handling & docs
Define clear error variant names (`GpError::MissingSelectKey { key }`, `GpError::InterpolationFailure { key, source }`, etc.) surfaced through ingestion errors. Document new GP architecture in README/docs to guide future optimizations.
