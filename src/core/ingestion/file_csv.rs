use super::Cast;
use super::Error;
use super::Extract;
use super::Ingestable;
use super::Sequence;
use std::collections::HashMap;

pub struct Csv {
    path: String,
    cast: Box<dyn Cast>,
    feature_pipelines: Vec<Extract>,
    target_pipelines: Vec<Extract>,
}

impl Csv {
    pub fn new(
        path: String,
        cast: Box<dyn Cast>,
        feature_pipelines: Vec<Extract>,
        target_pipelines: Vec<Extract>,
    ) -> Self {
        Csv {
            path,
            cast,
            feature_pipelines,
            target_pipelines,
        }
    }

    fn process_row(t: &mut [Extract], r: &HashMap<String, f32>) -> Result<Option<Vec<f32>>, Error> {
        let mut row = Vec::with_capacity(t.len());
        let mut has_none = false;
        for t in t.iter_mut() {
            let input = r.get(&t.source).map(Clone::clone);

            let mut output = t.interpolation.interpolate(input).map_err(|e| {
                Error::InterpolationError(
                    format!("Failed to interpolate '{}': {}", t.source, e).into(),
                )
            })?;

            for processor in &mut t.nodes {
                output = output.and_then(|v| processor.process(v));
            }

            // Only push if there is a value
            match output {
                Some(value) => {
                    if !has_none {
                        row.push(value)
                    }
                }
                None => has_none = true,
            }
        }

        if has_none { Ok(None) } else { Ok(Some(row)) }
    }
}

impl Ingestable for Csv {
    fn ingest(&mut self) -> Result<Sequence, Error> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&self.path)?;

        let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

        let mut seen_complete = false;
        let mut buf_features = Vec::new();
        let mut buf_targets = Vec::new();
        let mut buf_metadata = Vec::new();

        while let Some(record) = reader.records().next() {
            let record: HashMap<String, String> = headers
                .iter()
                .zip(record?.iter())
                .map(|(h, v)| (h.to_string(), v.to_string()))
                .collect();

            let casted = self.cast.cast(&record)?;

            let features = Self::process_row(&mut self.feature_pipelines, &casted)?;

            let targets = Self::process_row(&mut self.target_pipelines, &casted)?;

            match (features, targets, seen_complete) {
                (Some(features), Some(targets), _) => {
                    if !seen_complete {
                        seen_complete = true
                    }
                    buf_features.push(features);
                    buf_targets.push(targets);
                    buf_metadata.push(record)
                }
                (_, _, false) => continue,
                (_, _, true) => {
                    return Err(Error::NoneAfterSaturation);
                }
            }
        }

        let sequence = Sequence::new(buf_features, buf_targets, buf_metadata)?;
        Ok(sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::super::Cast;
    use super::super::Csv;
    use super::super::Error;
    use super::super::Extract;
    use super::super::Ingestable;
    use chrono::{Datelike, NaiveDate};
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;

    #[derive(Clone)]
    struct ExampleCast;

    impl Cast for ExampleCast {
        fn cast(&self, input: &HashMap<String, String>) -> Result<HashMap<String, f32>, Error> {
            let mut cast = HashMap::with_capacity(2);

            let year = input
                .get("Year")
                .and_then(|y| y.parse::<i32>().ok())
                .ok_or_else(|| Error::NotEnoughHistory("Missing or invalid Year".into()))?;

            let month = input
                .get("Month")
                .and_then(|m| m.parse::<u32>().ok())
                .ok_or_else(|| Error::NotEnoughHistory("Missing or invalid Month".into()))?;

            let day = input
                .get("Day")
                .and_then(|d| d.parse::<u32>().ok())
                .ok_or_else(|| Error::NotEnoughHistory("Missing or invalid Day".into()))?;

            let dow = NaiveDate::from_ymd_opt(year, month, day)
                .map(|d| (d.weekday().number_from_monday() - 1) as f32)
                .ok_or_else(|| Error::NotEnoughHistory("Invalid date".into()))?;

            cast.insert("dow".to_string(), dow);

            if let Some(value) = input.get("Value") {
                let trimmed = value.trim();
                // Treat "NA" as missing (don't insert into cast)
                if !trimmed.eq_ignore_ascii_case("NA") {
                    let v = trimmed
                        .parse::<f32>()
                        .map_err(|err| Error::CastingError(Box::new(err)))?;
                    cast.insert("Value".to_string(), v);
                }
            }

            Ok(cast)
        }

        fn clone_box(&self) -> Box<dyn Cast> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_csv_ingestion_with_day_of_week() {
        let csv_path = "/tmp/test_ingest.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        writeln!(file, "2025,1,15,5.5").unwrap();
        writeln!(file, "2025,1,16,6.2").unwrap();
        writeln!(file, "2025,1,17,7.1").unwrap();

        let mut csv = Csv::new(
            csv_path.to_string(),
            Box::new(ExampleCast),
            vec![Extract::new("dow")],
            vec![Extract::new("dow")],
        );

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count(), 3);

        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        assert_eq!(features.len(), 1);
        // First row: 2025-01-15 is Wednesday (dow=2)
        assert_eq!(features[0][0], 2.0);
        // Target with prediction_horizon=0 is at the same position: Wednesday (dow=2)
        assert_eq!(target[0], 2.0);
    }

    #[test]
    fn test_multiple_features_and_targets() {
        // Behavior: multiple Extract pipelines should produce multiple columns in features and targets.
        let csv_path = "/tmp/test_ingest_multi.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        writeln!(file, "2025,1,15,5.5").unwrap();
        writeln!(file, "2025,1,16,6.2").unwrap();

        let mut csv = Csv::new(
            csv_path.to_string(),
            Box::new(ExampleCast),
            vec![Extract::new("dow"), Extract::new("Value")],
            vec![Extract::new("Value"), Extract::new("dow")],
        );

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count(), 2);

        let (features, target) = sequence.get_item(0, 1, 0).unwrap();
        // Two features: dow and Value
        assert_eq!(features[0].len(), 2);
        assert_eq!(features[0][0], 2.0); // dow for Wednesday
        assert_eq!(features[0][1], 5.5); // Value
        // Two targets: Value and dow
        assert_eq!(target.len(), 2);
        assert_eq!(target[0], 5.5); // Value
        assert_eq!(target[1], 2.0); // dow
    }

    #[test]
    fn test_prediction_horizon_offsets_target() {
        // Behavior: with prediction_horizon > 0, the target should be offset to a future row.
        // prediction_horizon=1 means target is 1 row ahead.
        let csv_path = "/tmp/test_ingest_horizon.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        writeln!(file, "2025,1,15,5.5").unwrap();
        writeln!(file, "2025,1,16,6.2").unwrap();
        writeln!(file, "2025,1,17,7.1").unwrap();

        let mut csv = Csv::new(
            csv_path.to_string(),
            Box::new(ExampleCast),
            vec![Extract::new("Value")],
            vec![Extract::new("Value")],
        );

        let sequence = csv.ingest().unwrap();
        // With sequence_length=1 and prediction_horizon=1, we can have 1 complete item (3 - 1 - 1 = 1)
        let (features, target) = sequence.get_item(0, 1, 1).unwrap();
        // Features: row 0 (Value=5.5)
        assert_eq!(features[0][0], 5.5);
        // Target: row 1 (Value=6.2) because prediction_horizon=1 offsets by 1
        assert_eq!(target[0], 6.2);
    }

    #[test]
    fn test_interpolation_fills_gaps() {
        // Behavior: with linear interpolation, gaps in data can be filled using history.
        use crate::core::interpolation::{Interpolation, LinearInterpolator};

        let csv_path = "/tmp/test_ingest_interp.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "Year,Month,Day,Value").unwrap();
        // Values form a linear pattern: 10, 20, NA -> should interpolate to 30
        writeln!(file, "2025,1,15,10.0").unwrap();
        writeln!(file, "2025,1,16,20.0").unwrap();
        writeln!(file, "2025,1,17,NA").unwrap();

        let mut csv = Csv::new(
            csv_path.to_string(),
            Box::new(ExampleCast),
            vec![
                Extract::new("Value")
                    .with_interpolation(Interpolation::Linear(LinearInterpolator::new(2))),
            ],
            vec![Extract::new("dow")],
        );

        let sequence = csv.ingest().unwrap();
        assert_eq!(sequence.count(), 3);
        // Third row should be interpolated to 30.0
        let (features, _target) = sequence.get_item(2, 1, 0).unwrap();
        assert!((features[0][0] - 30.0).abs() < 1e-4);
    }
}
