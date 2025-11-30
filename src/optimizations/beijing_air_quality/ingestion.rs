use crate::core::ingestion::{Csv, Extract, Ingestable, ManySequences};
use crate::optimizations::beijing_air_quality::cast::BeijingCast;

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

pub async fn ingest(features: &[Extract], targets: &[Extract]) -> anyhow::Result<ManySequences> {
    let mut set = tokio::task::JoinSet::new();

    for path in PATHS {
        let features = features.to_vec();
        let targets = targets.to_vec();
        let path = path.to_string();
        set.spawn(async move {
            let mut ingestable = Csv::new(path, Box::new(BeijingCast), features, targets);
            ingestable.ingest()
        });
    }

    let mut sequences = Vec::new();
    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok(sequence)) => sequences.push(sequence),
            Ok(Err(e)) => return Err(e.into()),
            Err(e) => return Err(e.into()),
        }
    }

    Ok(ManySequences::new(sequences))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::interpolation::{Interpolation, LinearInterpolator};

    #[tokio::test]
    async fn test_ingest_with_linear_interpolation_window_1() {
        let features = &[
            Extract::new("PM2.5")
                .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            Extract::new("PM10")
                .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
        ];

        let targets = &[Extract::new("O3")
            .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1)))];

        let result = ingest(features, targets).await;
        assert!(result.is_ok());

        let sequences = result.unwrap();
        assert!(sequences.len(1, 0) > 0);
    }

    #[tokio::test]
    async fn test_ingest_multiple_extracts() {
        let features = &[
            Extract::new("PM2.5")
                .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            Extract::new("TEMP")
                .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            Extract::new("WSPM")
                .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
        ];

        let targets = &[Extract::new("PM10")
            .with_interpolation(Interpolation::Linear(LinearInterpolator::new(1)))];

        let result = ingest(features, targets).await;
        assert!(result.is_ok());
    }
}
