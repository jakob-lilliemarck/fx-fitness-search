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

/// Extract a required value from a CSV row and parse it.
/// Treats the specified missing_value string as missing data.
pub fn extract_required<T: std::str::FromStr>(
    column: &str,
    input: &HashMap<String, String>,
    missing_value: &str,
) -> Result<T, Error>
where
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let s = input
        .get(column)
        .ok_or_else(|| Error::NotEnoughHistory(format!("Missing required column: {}", column)))?;

    let trimmed = s.trim();

    // Treat the specified missing_value as missing data
    if trimmed.eq_ignore_ascii_case(missing_value) {
        return Err(Error::NotEnoughHistory(format!(
            "{} has missing value",
            column
        )));
    }

    trimmed
        .parse::<T>()
        .map_err(|err| Error::CastingError(Box::new(err)))
}

/// Extract an optional value from a CSV row and parse it.
/// Treats the specified missing_value string and missing columns as None.
pub fn extract_optional<T: std::str::FromStr>(
    column: &str,
    input: &HashMap<String, String>,
    missing_value: &str,
) -> Result<Option<T>, Error>
where
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let s = match input.get(column) {
        Some(s) => s,
        None => return Ok(None),
    };

    let trimmed = s.trim();

    // Treat the specified missing_value as missing data
    if trimmed.eq_ignore_ascii_case(missing_value) {
        return Ok(None);
    }

    trimmed
        .parse::<T>()
        .map(Option::Some)
        .map_err(|err| Error::CastingError(Box::new(err)))
}

mod extract;
mod file_csv;
mod sequence;

pub use extract::Extract;
pub use file_csv::Csv;
pub use sequence::{ManySequences, Sequence};
