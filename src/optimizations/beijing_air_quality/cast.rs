use crate::core::ingestion::{self, Cast, extract_optional, extract_required};
use chrono::{Datelike, NaiveDate};
use geo::Point;
use geo::algorithm::centroid::Centroid;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Could not find value for '{0}'")]
    NotFound(String),
    #[error("Could not parse value as integer: '{got}'")]
    Parse {
        source: Box<dyn std::error::Error + Send + Sync>,
        got: String,
    },
    #[error("Centroid returned None")]
    Centroid,
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

#[derive(Debug, Hash, PartialEq, Eq)]
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

const STATIONS: [Station; 12] = [
    Station::Aotizhongxin,
    Station::Changping,
    Station::Dingling,
    Station::Dongsi,
    Station::Guanyuan,
    Station::Gucheng,
    Station::Huairou,
    Station::Nongzhanguan,
    Station::Shunyi,
    Station::Tiantan,
    Station::Wanliu,
    Station::Wanshouxigong,
];

impl Station {
    /// Returns the centroid of all stations in decimal degrees (lon, lat)
    fn centroid() -> Result<(f64, f64), Error> {
        let mut points = Vec::new();
        for station in STATIONS {
            let (lon, lat) = station.position()?;
            points.push(Point::new(lon, lat));
        }
        let multi_point = geo::MultiPoint(points);
        multi_point
            .centroid()
            .map(|p| (p.x(), p.y()))
            .ok_or_else(|| Error::Centroid)
    }

    fn position(&self) -> Result<(f64, f64), Error> {
        // reference:
        // https://pmc.ncbi.nlm.nih.gov/articles/PMC9675857/
        let coords = match self {
            Self::Aotizhongxin => "39°59'6.562\" N 116°24'3.028\" E",
            Self::Changping => "40°13'15.427\" N 116°13'51.722\" E",
            Self::Dingling => "40°17'41.435\" N 116°13'32.732\" E",
            Self::Dongsi => "39°55'46.423\" N 116°24'55.862\" E",
            Self::Guanyuan => "39°55'57.986\" N 116°21'24.548\" E",
            Self::Gucheng => "39°54'47.041\" N 116°11'6.130\" E",
            Self::Huairou => "40°19'1.211\" N 116°37'55.106\" E",
            Self::Nongzhanguan => "39°56'25.872\" N 116°27'54.468\" E",
            Self::Shunyi => "40°07'37.2\" N 116°39'18.0\" E",
            Self::Tiantan => "39°52'54.887\" N 116°24'38.984\" E",
            Self::Wanliu => "39°58'1.758\" N 116°17'53.063\" E",
            Self::Wanshouxigong => "39°54'34.171\" N 116°17'40.420\" E",
        };
        latlon::parse(coords)
            .map(|p| (p.x(), p.y()))
            .map_err(|err| Error::Parse {
                source: Box::new(err),
                got: coords.to_string(),
            })
    }

    /// Returns a vector (dx, dy) from the centroid to this station in decimal degrees
    ///
    /// - dx is the longitude difference (positive = east)
    /// - dy is the latitude difference (positive = north)
    fn vector(&self, center_lon: f64, center_lat: f64) -> Result<(f64, f64), Error> {
        let (lon, lat) = self.position()?;
        let dx = lon - center_lon;
        let dy = lat - center_lat;
        Ok((dx, dy))
    }

    /// Computes the maximum Euclidean distance (in deg) from centroid to any station
    fn max_distance(center_lon: f64, center_lat: f64) -> Result<f64, Error> {
        let mut max_dist = 0.0;
        for station in STATIONS {
            let (dx, dy) = station.vector(center_lon, center_lat)?;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > max_dist {
                max_dist = dist;
            }
        }
        Ok(max_dist)
    }

    /// Returns a normalized vector where the farthest station has norm 1.0
    pub fn normalized_vector(&self, center_lon: f64, center_lat: f64) -> Result<(f64, f64), Error> {
        let (dx, dy) = self.vector(center_lon, center_lat)?;
        let max_dist = Self::max_distance(center_lon, center_lat)?;
        if max_dist == 0.0 {
            return Ok((0.0, 0.0));
        }
        Ok((dx / max_dist, dy / max_dist))
    }
}

impl FromStr for Station {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
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
            _ => Err(Error::Station(s.to_string())),
        }
    }
}

static LOCATIONS: OnceLock<HashMap<Station, (f64, f64)>> = OnceLock::new();

fn get_location<'a>(station: Station) -> Option<&'a (f64, f64)> {
    let locations = LOCATIONS.get_or_init(|| {
        let mut stations = HashMap::with_capacity(12);

        let (center_lon, center_lat) =
            Station::centroid().expect("Could not compute centroid of stations");

        for station in STATIONS {
            let v = station
                .normalized_vector(center_lon, center_lat)
                .expect("Could not compute normalized vector location");

            stations.insert(station, v);
        }

        stations
    });

    locations.get(&station)
}

#[derive(Clone)]
pub struct BeijingCast;

impl Cast for BeijingCast {
    fn cast(
        &self,
        input: &HashMap<String, String>,
    ) -> Result<HashMap<String, f32>, ingestion::Error> {
        let mut cast = HashMap::with_capacity(15);
        const MISSING_VALUE: &str = "NA";

        // Get the station vector
        let station = extract_required::<Station>("station", input, MISSING_VALUE)?;
        let (pos_x, pos_y) = get_location(station).expect("Could not get location of station");
        cast.insert("pos_x".to_string(), pos_x.to_owned() as f32);
        cast.insert("pos_y".to_string(), pos_y.to_owned() as f32);

        // Extract and extrapolare all time related columns
        let y = extract_required::<i32>("year", input, MISSING_VALUE)?;
        let m = extract_required::<u32>("month", input, MISSING_VALUE)?;
        let d = extract_required::<u32>("day", input, MISSING_VALUE)?;
        let h = extract_required::<u32>("hour", input, MISSING_VALUE)?;
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
            if let Some(value) = extract_optional(key, input, MISSING_VALUE)? {
                cast.insert(key.to_string(), value);
            }
        }

        // Extract, cast and insert wind direction
        if let Some(wd_str) = input.get("wd") {
            let trimmed = wd_str.trim();
            // Only try to parse if not NA
            if !trimmed.eq_ignore_ascii_case("NA") {
                if let Some(value) = WindDirection::try_from(trimmed).ok() {
                    cast.insert("wd".to_string(), value.into());
                }
            }
        }

        Ok(cast)
    }

    fn clone_box(&self) -> Box<dyn Cast> {
        Box::new(self.clone())
    }
}

pub enum Optimizable {
    Pm25,
    Pm10,
    So2,
    No2,
    Co,
    O3,
    Temp,
    Pres,
    Dewp,
    Rain,
    Wspm,
    Wd,
}

#[derive(Debug, thiserror::Error)]
pub enum OptimizableError {
    #[error("Unknown optimizable key: '{0}'")]
    Unknown(String),
}

impl TryFrom<&str> for Optimizable {
    type Error = OptimizableError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "PM2.5" => Ok(Optimizable::Pm25),
            "PM10" => Ok(Optimizable::Pm10),
            "SO2" => Ok(Optimizable::So2),
            "NO2" => Ok(Optimizable::No2),
            "CO" => Ok(Optimizable::Co),
            "O3" => Ok(Optimizable::O3),
            "TEMP" => Ok(Optimizable::Temp),
            "PRES" => Ok(Optimizable::Pres),
            "DEWP" => Ok(Optimizable::Dewp),
            "RAIN" => Ok(Optimizable::Rain),
            "WSPM" => Ok(Optimizable::Wspm),
            "wd" => Ok(Optimizable::Wd),
            _ => Err(OptimizableError::Unknown(value.to_string())),
        }
    }
}

use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

impl fmt::Display for Optimizable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Optimizable::Pm25 => "PM2.5",
            Optimizable::Pm10 => "PM10",
            Optimizable::So2 => "SO2",
            Optimizable::No2 => "NO2",
            Optimizable::Co => "CO",
            Optimizable::O3 => "O3",
            Optimizable::Temp => "TEMP",
            Optimizable::Pres => "PRES",
            Optimizable::Dewp => "DEWP",
            Optimizable::Rain => "RAIN",
            Optimizable::Wspm => "WSPM",
            Optimizable::Wd => "wd",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_station_position_aotizhongxin() {
        let station = Station::Aotizhongxin;
        let (lon, lat) = station.position().expect("Failed to parse Aotizhongxin");
        let expected_lat = 39.0 + 59.0 / 60.0 + 6.562 / 3600.0;
        let expected_lon = 116.0 + 24.0 / 60.0 + 3.028 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_changping() {
        let station = Station::Changping;
        let (lon, lat) = station.position().expect("Failed to parse Changping");
        let expected_lat = 40.0 + 13.0 / 60.0 + 15.427 / 3600.0;
        let expected_lon = 116.0 + 13.0 / 60.0 + 51.722 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_dingling() {
        let station = Station::Dingling;
        let (lon, lat) = station.position().expect("Failed to parse Dingling");
        let expected_lat = 40.0 + 17.0 / 60.0 + 41.435 / 3600.0;
        let expected_lon = 116.0 + 13.0 / 60.0 + 32.732 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_dongsi() {
        let station = Station::Dongsi;
        let (lon, lat) = station.position().expect("Failed to parse Dongsi");
        let expected_lat = 39.0 + 55.0 / 60.0 + 46.423 / 3600.0;
        let expected_lon = 116.0 + 24.0 / 60.0 + 55.862 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_guanyuan() {
        let station = Station::Guanyuan;
        let (lon, lat) = station.position().expect("Failed to parse Guanyuan");
        let expected_lat = 39.0 + 55.0 / 60.0 + 57.986 / 3600.0;
        let expected_lon = 116.0 + 21.0 / 60.0 + 24.548 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_gucheng() {
        let station = Station::Gucheng;
        let (lon, lat) = station.position().expect("Failed to parse Gucheng");
        let expected_lat = 39.0 + 54.0 / 60.0 + 47.041 / 3600.0;
        let expected_lon = 116.0 + 11.0 / 60.0 + 6.130 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_huairou() {
        let station = Station::Huairou;
        let (lon, lat) = station.position().expect("Failed to parse Huairou");
        let expected_lat = 40.0 + 19.0 / 60.0 + 1.211 / 3600.0;
        let expected_lon = 116.0 + 37.0 / 60.0 + 55.106 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_nongzhanguan() {
        let station = Station::Nongzhanguan;
        let (lon, lat) = station.position().expect("Failed to parse Nongzhanguan");
        let expected_lat = 39.0 + 56.0 / 60.0 + 25.872 / 3600.0;
        let expected_lon = 116.0 + 27.0 / 60.0 + 54.468 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_shunyi() {
        let station = Station::Shunyi;
        let (lon, lat) = station.position().expect("Failed to parse Shunyi");
        let expected_lat = 40.0 + 7.0 / 60.0 + 37.2 / 3600.0;
        let expected_lon = 116.0 + 39.0 / 60.0 + 18.0 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_tiantan() {
        let station = Station::Tiantan;
        let (lon, lat) = station.position().expect("Failed to parse Tiantan");
        let expected_lat = 39.0 + 52.0 / 60.0 + 54.887 / 3600.0;
        let expected_lon = 116.0 + 24.0 / 60.0 + 38.984 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_wanliu() {
        let station = Station::Wanliu;
        let (lon, lat) = station.position().expect("Failed to parse Wanliu");
        let expected_lat = 39.0 + 58.0 / 60.0 + 1.758 / 3600.0;
        let expected_lon = 116.0 + 17.0 / 60.0 + 53.063 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_station_position_wanshouxigong() {
        let station = Station::Wanshouxigong;
        let (lon, lat) = station.position().expect("Failed to parse Wanshouxigong");
        let expected_lat = 39.0 + 54.0 / 60.0 + 34.171 / 3600.0;
        let expected_lon = 116.0 + 17.0 / 60.0 + 40.420 / 3600.0;
        assert!((lon - expected_lon).abs() < 1e-9, "lon mismatch");
        assert!((lat - expected_lat).abs() < 1e-9, "lat mismatch");
    }

    #[test]
    fn test_centroid_computation() {
        let (center_lon, center_lat) = Station::centroid().expect("Failed to compute centroid");
        // Centroid should be roughly in the middle of Beijing's station network
        // Approximate bounds: lon 116.1 - 116.65, lat 39.52 - 40.32
        assert!(
            center_lon > 116.1 && center_lon < 116.7,
            "centroid lon out of expected range"
        );
        assert!(
            center_lat > 39.5 && center_lat < 40.4,
            "centroid lat out of expected range"
        );
    }

    #[test]
    fn test_vector_aotizhongxin() {
        let (center_lon, center_lat) = Station::centroid().expect("Could not get centroid");
        let station = Station::Aotizhongxin;
        let (dx, dy) = station
            .vector(center_lon, center_lat)
            .expect("Failed to compute vector");
        // Aotizhongxin is roughly in the center, so vector should be small
        assert!(
            dx.abs() < 0.5 && dy.abs() < 0.5,
            "Aotizhongxin should be near centroid"
        );
    }

    #[test]
    fn test_vector_relative_positions() {
        let (center_lon, center_lat) = Station::centroid().expect("Could not get centroid");
        let station1 = Station::Aotizhongxin;
        let station2 = Station::Shunyi;
        let (dx1, dy1) = station1
            .vector(center_lon, center_lat)
            .expect("Failed to compute vector for Aotizhongxin");
        let (dx2, dy2) = station2
            .vector(center_lon, center_lat)
            .expect("Failed to compute vector for Shunyi");
        // Shunyi is northeast: should have positive dx and dy
        assert!(dx2 > 0.0, "Shunyi should be east of centroid");
        assert!(dy2 > 0.0, "Shunyi should be north of centroid");
        // Vectors should differ
        let diff = ((dx2 - dx1).powi(2) + (dy2 - dy1).powi(2)).sqrt();
        assert!(
            diff > 0.1,
            "station vectors should be significantly different"
        );
    }

    #[test]
    fn test_normalized_vector_range() {
        let (center_lon, center_lat) = Station::centroid().expect("Could not get centroid");
        let station = Station::Aotizhongxin;
        let (nx, ny) = station
            .normalized_vector(center_lon, center_lat)
            .expect("Failed to compute normalized vector");
        // Both components should be in [-1, 1]
        assert!(
            nx >= -1.0 && nx <= 1.0,
            "normalized dx should be in [-1, 1]"
        );
        assert!(
            ny >= -1.0 && ny <= 1.0,
            "normalized dy should be in [-1, 1]"
        );
        // Norm should be <= 1
        let norm = (nx * nx + ny * ny).sqrt();
        assert!(norm <= 1.0 + 1e-9, "normalized vector norm should be <= 1");
    }

    #[test]
    fn test_normalized_vector_scaling() {
        let (center_lon, center_lat) = Station::centroid().expect("Could not get centroid");
        // Find the farthest station (should have norm close to 1.0)
        let mut max_norm = 0.0;
        for station in STATIONS {
            let (nx, ny) = station
                .normalized_vector(center_lon, center_lat)
                .expect("Failed to compute normalized vector");
            let norm = (nx * nx + ny * ny).sqrt();
            if norm > max_norm {
                max_norm = norm;
            }
        }
        // The farthest station should have norm very close to 1.0
        assert!(
            (max_norm - 1.0).abs() < 1e-9,
            "farthest station should have norm 1.0"
        );
    }
}
