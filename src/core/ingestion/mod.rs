use std::collections::HashMap;
use thiserror;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Could not read file: {0}")]
    ReadError(#[from] csv::Error),
    #[error("Interpolation did not have enough hisotry {0}")]
    NotEnoughHistory(String),
    #[error("Encountered None value after saturation")]
    NoneAfterSaturation,
    #[error("Factor out of bounds")]
    FactorOutOfBounds,
    #[error("CastingError: {0}")]
    CastingError(Box<dyn std::error::Error + Send + Sync>),
    #[error("InterpolationError: {0}")]
    InterpolationError(Box<dyn std::error::Error + Send + Sync>),
    #[error("ProcessingError: {0}")]
    ProcessingError(Box<dyn std::error::Error + Send + Sync>),
    #[error("SequenceError: {0}")]
    SequenceError(#[from] sequence::Error),
}

pub trait Ingestable {
    fn ingest(&mut self) -> Result<Sequence, Error>;
}

pub trait Cast: Send {
    fn cast(&self, input: &HashMap<String, String>) -> Result<HashMap<String, f32>, Error>;
    fn clone_box(&self) -> Box<dyn Cast>;
}

mod extract;
mod file_csv;
mod sequence;

pub use extract::Extract;
pub use file_csv::Csv;
pub use sequence::{ManySequences, Sequence};
