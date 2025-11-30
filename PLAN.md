# Core Module Improvement Plan

## Overview
This document tracks ongoing improvements to `./src/core` covering test coverage, code structure, correctness, and readability.

**Last Review Date:** 2025-11-27

## Current Structure

```
src/core/
├── mod.rs
├── batcher.rs                ✓ (tested)
├── dataset.rs                ✓ (tested)
├── model.rs                  ✓ (tested)
├── interpolation.rs          ✓ (tested)
├── train.rs                  ✗ (NO TESTS)
├── train_config.rs           ✗ (NO TESTS)
├── inference.rs              ✗ (NO TESTS)
├── ingestion/
│   ├── mod.rs                (traits + errors)
│   ├── extract.rs            (transformation pipeline)
│   ├── sequence.rs           ✓ (40+ tests)
│   └── file_csv.rs           ✓ (5 integration tests)
└── preprocessing/
    ├── mod.rs                (exports)
    ├── node.rs               ✓ (composite tests)
    ├── ema.rs                (no unit tests)
    ├── cos.rs                (no unit tests)
    ├── sin.rs                (no unit tests)
    ├── roc.rs                (no unit tests)
    ├── std.rs                (no unit tests)
    └── zscore.rs             (no unit tests)
```

## Assessment

### ✓ Excellent Structure & Tests

**ingestion/** - Well-refactored with clear concerns:
* mod.rs defines Cast/Ingestable traits (glue layer)
* extract.rs: transformation pipeline (reusable)
* sequence.rs: Sequence + ManySequences with 40+ comprehensive tests
* file_csv.rs: CSV ingestion with 5 integration tests

**preprocessing/** - Cleanly split with each transformer isolated:
* Each transformer (Ema, Cos, Sin, Roc, Std, ZScore) in its own file
* node.rs contains Node enum + composite tests
* Tests moved to node.rs (good for integration tests)
* Issue: Individual transformer files lack unit tests

**Other tested modules** - batcher, dataset, model, interpolation all have good test coverage

### ✗ CRITICAL GAPS: Missing Tests

**1. train.rs (179 lines)**
* Single large function (train_sync)
* Handles: optimizer initialization, epoch loop, batch processing, gradient updates, validation loop, early stopping, model saving
* No tests whatsoever
* Complex logic critical to correctness: loss calculation, backward pass, optimizer step, early stopping condition

**2. train_config.rs (45 lines)**
* Serialization/deserialization (save/load)
* No tests for round-trip or error conditions
* Critical for model persistence

**3. inference.rs (73 lines)**
* Model loading and prediction interface
* No tests for load() or predict() functions
* Critical user-facing API

## Code Quality Issues

### train.rs - Correctness Concerns

| Issue | Line(s) | Impact | Fix |
|-------|---------|--------|-----|
| Hardcoded patience=10 | 47 | Not configurable for different datasets | Extract to parameter or config |
| Hardcoded weight decay=5e-4 | 33 | Not tunable | Extract to parameter |
| Hardcoded grad clip=1.0 | 34 | Not tunable | Extract to parameter |
| No dataset validation | N/A | Could fail silently | Add checks before processing |
| MSE always used | 79 | Only loss function supported | Make configurable |

**Specific concerns:**
1. Loss calculation: `(outputs - batch.targets).powf_scalar(2.0).mean()` - assumes MSE always appropriate?
2. Line 91: `model = optimizer.step(...)` mutates model in place; behavior depends on Burn's optimizer implementation
3. Early stopping hardcoded to patience=10; not configurable
4. No validation that datasets have items before processing
5. No handling if batcher returns empty batch (caught by line 66-68 but still edge case)

### train_config.rs - Validation Issues

| Issue | Line(s) | Impact | Fix |
|-------|---------|--------|-----|
| No field validation | 16-30 | Can create invalid configs | Add validation in `new()` |
| No range checks | N/A | Zero-sized sequences/hidden layers possible | Add assert/error |
| dead_code on load() | 38 | Suggests incomplete usage | Remove or verify usage |

**Specific issues:**
1. No validation on field values (hidden_size, sequence_length could be 0)
2. save/load use ? operator but don't validate on load
3. `#[allow(dead_code)]` on load() suggests it might not be used everywhere

### inference.rs - Quality Issues

1. Line 52: Placeholder target vec filled with zeros; what if model expects different target layout?
2. Line 25: Hardcoded config path format `"{}.config.json"`; should this be configurable?
3. Line 26: Error message leaks internal path details

### preprocessing/ - Minor Issue

* Individual transformer files (ema.rs, cos.rs, etc.) have NO unit tests
* Only node.rs has composite tests
* Hard to test individual transformers in isolation without going through Node enum

## Readability & Organization

**Positive:**
* Module names are clear (preprocessing, ingestion, train, inference)
* Dependencies are explicit and minimal
* ingestion/sequence.rs tests are well-documented with behavior comments
* preprocessing/ structure makes it easy to add new transformers

**Negative:**
* train.rs function is dense (179 lines); difficult to follow all phases
* preprocessing transformer files have no tests, only composite tests in node.rs
* train.rs has hardcoded hyperparameters (patience=10, weight_decay=5e-4, grad_clip=1.0)

## Recommendations (Priority Order)

### Priority 1 - Add Critical Tests
**Effort:** HIGH | **Impact:** HIGH | **Urgency:** IMMEDIATE

Target completion: Before next integration/deployment

- [ ] **train_config.rs** (estimated: 1-2 hours)
  - Save/load round-trip test
  - Field validation tests
  - Error condition tests
  
- [ ] **inference.rs** (estimated: 2-3 hours)
  - load() with valid paths
  - load() error handling
  - predict() basic functionality
  - Tensor conversion correctness
  
- [ ] **train.rs** (estimated: 4-6 hours)
  - Requires Burn backend mocking or NdArray backend
  - Single epoch training
  - Early stopping logic
  - Loss calculation
  - Model saving

### Priority 2 - Correctness Fixes
**Effort:** MEDIUM | **Impact:** HIGH | **Urgency:** HIGH

- [ ] **train.rs** - Extract hardcoded hyperparameters (patience, weight_decay, grad_clip)
  - Option A: Add to TrainConfig struct
  - Option B: Add to train_sync function signature
  - Decision: Recommend Option A for persistence

- [ ] **train.rs** - Add dataset validation
  - Check training dataset not empty before processing
  - Check validation dataset not empty before processing
  - Return error if validation datasets are empty

- [ ] **train_config.rs** - Add field validation
  - hidden_size > 0
  - sequence_length > 0
  - prediction_horizon >= 0

- [ ] **preprocessing/** - Add unit tests to individual transformer files
  - Each transformer file should have its own test module
  - Test normal cases and edge cases (zero values, extreme values)

### Priority 3 - Readability Improvements
**Effort:** MEDIUM | **Impact:** MEDIUM | **Urgency:** MEDIUM

- [ ] **train.rs** - Extract helper functions for readability
  - `run_training_epoch()` → returns (total_loss, num_batches)
  - `run_validation_epoch()` → returns (total_loss, num_batches)
  - `check_early_stopping()` → returns bool
  - `maybe_save_model()` → returns Result<()>

- [ ] **preprocessing/** - Document test structure
  - Add module-level comment explaining why tests are in node.rs
  - Add individual unit tests to each transformer file

### Priority 4 - Architecture Improvements
**Effort:** HIGH | **Impact:** MEDIUM | **Urgency:** LOW

Consider for future improvements:

- [ ] Extract EarlyStoppingTracker into separate module
  - Encapsulate patience, best_loss, epochs_without_improvement
  - Makes early stopping testable in isolation

- [ ] Make loss function configurable
  - Currently hardcoded to MSE
  - Create LossFunction trait?

- [ ] Make optimizer config parameterizable
  - Currently hardcoded AdamConfig with specific hyperparameters
  - Allow users to customize

## Testing Strategy

### For train.rs
Use Burn's NdArray backend for unit tests (CPU-only, deterministic)
```rust
#[test]
fn test_single_epoch_training() {
    type B = burn::backend::ndarray::NdArray;
    // Create mock dataset, model, run single epoch
}
```

### For train_config.rs
Simple file I/O tests with temporary files
```rust
#[test]
fn test_save_load_roundtrip() {
    let config = TrainConfig::new(...);
    config.save("/tmp/test_config.json").unwrap();
    let loaded = TrainConfig::load("/tmp/test_config.json").unwrap();
    assert_eq!(config, loaded); // May need PartialEq impl
}
```

### For inference.rs
Mock or use real model files from test fixtures
```rust
#[test]
fn test_inference_predict() {
    // Load test model + config
    // Call predict with dummy sequence
    // Verify output dimensions
}
```

### For preprocessing transformers
Unit tests in each file:
```rust
// In ema.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_ema_warmup() { }
}
```

## Notes & Decisions

- **Sequence + ManySequences kept together:** Good decision—they're tightly coupled
- **Cast trait in ingestion/mod.rs:** Appropriate boundary definition
- **preprocessing/ split:** Each transformer isolated, good for extensibility
- **Tests in preprocessing/node.rs:** Currently only composite tests; need individual unit tests

## Tracking

| Task | Owner | Status | Target Date | Notes |
|------|-------|--------|-------------|-------|
| Add train_config tests | Jakob | ✅ DONE | 2025-11-27 | 10 comprehensive tests: save/load round-trip, JSON validation, error handling |
| Add inference tests | TBD | ⏳ TODO | TBD | Medium complexity |
| Add train tests | TBD | ⏳ TODO | TBD | High complexity—Burn mocking needed |
| Extract train.rs hyperparameters | TBD | ⏳ TODO | TBD | Depends on Priority 1 tests passing |
| Add preprocessing unit tests | TBD | ⏳ TODO | TBD | After train.rs tests |
| Refactor train.rs helpers | TBD | ⏳ TODO | TBD | Last priority—readability only |

---

## Next Steps

1. Start with train_config.rs tests (quick wins build momentum)
2. Move to inference.rs tests
3. Tackle train.rs with Burn backend mocking strategy
4. Address correctness fixes in parallel with testing
5. Follow up with readability improvements
