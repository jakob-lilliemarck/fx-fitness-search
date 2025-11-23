use crate::core::ingestion::{self, Cast};
use chrono::{Datelike, NaiveDate};
use std::{collections::HashMap, str::FromStr};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Could not find value for '{0}'")]
    NotFound(String),
    #[error("Could not parse value as integer: '{got}'")]
    Parse {
        source: Box<dyn std::error::Error + Send + Sync>,
        got: String,
    },
    #[error("Could not parse value as WindDirection '{0}'")]
    WindDirection(String),
    #[error("Could not parse value as Station '{0}'")]
    Station(String),
}

impl From<Error> for ingestion::Error {
    fn from(value: Error) -> Self {
        ingestion::Error::CastingError(Box::new(value))
    }
}

#[derive(Debug, Clone)]
pub enum WindDirection {
    N,
    NNE,
    NE,
    ENE,
    E,
    ESE,
    SE,
    SSE,
    S,
    SSW,
    SW,
    WSW,
    W,
    WNW,
    NW,
    NNW,
}

impl From<WindDirection> for f32 {
    fn from(value: WindDirection) -> Self {
        match value {
            WindDirection::N => 0.0,
            WindDirection::NNE => 22.5,
            WindDirection::NE => 45.0,
            WindDirection::ENE => 67.5,
            WindDirection::E => 90.0,
            WindDirection::ESE => 112.5,
            WindDirection::SE => 135.0,
            WindDirection::SSE => 157.5,
            WindDirection::S => 180.0,
            WindDirection::SSW => 202.5,
            WindDirection::SW => 225.0,
            WindDirection::WSW => 247.5,
            WindDirection::W => 270.0,
            WindDirection::WNW => 292.5,
            WindDirection::NW => 315.0,
            WindDirection::NNW => 337.5,
        }
    }
}

impl TryFrom<&str> for WindDirection {
    type Error = Error;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "N" => Ok(WindDirection::N),
            "NNE" => Ok(WindDirection::NNE),
            "NE" => Ok(WindDirection::NE),
            "ENE" => Ok(WindDirection::ENE),
            "E" => Ok(WindDirection::E),
            "ESE" => Ok(WindDirection::ESE),
            "SE" => Ok(WindDirection::SE),
            "SSE" => Ok(WindDirection::SSE),
            "S" => Ok(WindDirection::S),
            "SSW" => Ok(WindDirection::SSW),
            "SW" => Ok(WindDirection::SW),
            "WSW" => Ok(WindDirection::WSW),
            "W" => Ok(WindDirection::W),
            "WNW" => Ok(WindDirection::WNW),
            "NW" => Ok(WindDirection::NW),
            "NNW" => Ok(WindDirection::NNW),
            other => Err(Error::WindDirection(other.to_string())),
        }
    }
}

#[derive(Debug)]
pub enum Station {
    Aotizhongxin,
    Changping,
    Dingling,
    Dongsi,
    Guanyuan,
    Gucheng,
    Huairou,
    Nongzhanguan,
    Shunyi,
    Tiantan,
    Wanliu,
    Wanshouxigong,
}

// FIXME:
// We should define coorindates for each station.
// 1. Find the centroid using caretesian coordinates or LAT/LON
// 2. Define polar coordinates or a vector for each station using the centroid
//
// consider to implement this as a stateful preprocessor instead. that way we could keep a cache of centroid and a lookup table
impl Station {
    // FIXME:
    /// Returns centroid of all stations as cartesian or LAT/LON coordinates
    fn centroid() -> (f32, f32) {
        todo!()
    }

    // FIXME:
    /// Returns the position of the station as cartesian or LAT/LON coodinates
    fn position(&self) -> (f32, f32) {
        todo!()
    }

    // FIXME:
    /// Returns a non-normalized non-scaled vector from the centroid to the station
    fn vector(&self) -> (f32, f32) {
        todo!()
    }
}

impl TryFrom<&str> for Station {
    type Error = Error;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "Aotizhongxin" => Ok(Station::Aotizhongxin),
            "Changping" => Ok(Station::Changping),
            "Dingling" => Ok(Station::Dingling),
            "Dongsi" => Ok(Station::Dongsi),
            "Guanyuan" => Ok(Station::Guanyuan),
            "Gucheng" => Ok(Station::Gucheng),
            "Huairou" => Ok(Station::Huairou),
            "Nongzhanguan" => Ok(Station::Nongzhanguan),
            "Shunyi" => Ok(Station::Shunyi),
            "Tiantan" => Ok(Station::Tiantan),
            "Wanliu" => Ok(Station::Wanliu),
            "Wanshouxigong" => Ok(Station::Wanshouxigong),
            _ => Err(Error::Station(value.to_string())),
        }
    }
}

fn extract_required<T: FromStr>(column: &str, input: &HashMap<String, String>) -> Result<T, Error>
where
    <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let s = input
        .get(column)
        .ok_or(Error::NotFound(column.to_string()))?;

    s.trim().parse::<T>().map_err(|err| Error::Parse {
        source: Box::new(err),
        got: s.to_string(),
    })
}

fn extract_optional<T: FromStr>(
    column: &str,
    input: &HashMap<String, String>,
) -> Result<Option<T>, Error>
where
    <T as FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let s = match input.get(column) {
        Some(s) => s,
        None => return Ok(None),
    };

    s.trim()
        .parse::<T>()
        .map(Option::Some)
        .map_err(|err| Error::Parse {
            source: Box::new(err),
            got: s.to_string(),
        })
}

fn extract_and_insert_optional(
    column: &str,
    input: &HashMap<String, String>,
    output: &mut HashMap<String, f32>,
) -> Result<(), Error> {
    if let Some(value) = extract_optional(column, input)? {
        output.insert(column.to_string(), value);
    }

    Ok(())
}

#[derive(Clone)]
pub struct BeijingCast;

impl Cast for BeijingCast {
    fn cast(
        &self,
        input: &HashMap<String, String>,
    ) -> Result<HashMap<String, f32>, ingestion::Error> {
        let mut cast = HashMap::with_capacity(15);

        // Extract and extrapolare all time related columns
        let y = extract_required::<i32>("Year", input)?;
        let m = extract_required::<u32>("Month", input)?;
        let d = extract_required::<u32>("Day", input)?;
        let h = extract_required::<u32>("Hour", input)?;
        let day_of_week = NaiveDate::from_ymd_opt(y, m, d)
            .map(|dt| dt.weekday().number_from_monday() - 1)
            .ok_or_else(|| ingestion::Error::NotEnoughHistory("Invalid date".into()))?;
        cast.insert("hour".to_string(), h as f32);
        cast.insert("day_of_week".to_string(), day_of_week as f32);
        cast.insert("month".to_string(), m as f32);

        // Extract and insert all numerical columns
        for key in [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        ] {
            extract_and_insert_optional(key, input, &mut cast)?;
        }

        // Extract, cast and insert wind direction
        if let Some(value) = input
            .get("wd")
            .map(|s| WindDirection::try_from(s.as_str()))
            .transpose()?
        {
            cast.insert("wind_direction".to_string(), value.into());
        }

        Ok(cast)
    }

    fn clone_box(&self) -> Box<dyn Cast> {
        Box::new(self.clone())
    }
}
