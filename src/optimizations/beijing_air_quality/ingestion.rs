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

pub async fn ingest(
    feature_pipelines: Vec<Extract>,
    target_pipelines: Vec<Extract>,
) -> anyhow::Result<ManySequences> {
    let mut set = tokio::task::JoinSet::new();

    for path in PATHS {
        let pipelines_f = feature_pipelines.clone();

        let pipelines_t = target_pipelines.clone();

        let path = path.to_string();

        set.spawn(async move {
            let mut ingestable = Csv::new(path, Box::new(BeijingCast), pipelines_f, pipelines_t);
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
    use crate::core::interpolation::{LinearInterpolator, Interpolation};

    #[tokio::test]
    async fn test_ingest_with_linear_interpolation_window_1() {
        let feature_pipelines = vec![
            Extract::new("PM2.5").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            Extract::new("PM10").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
        ];
        let target_pipelines = vec![
            Extract::new("O3").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
        ];

        let result = ingest(feature_pipelines, target_pipelines).await;
        assert!(result.is_ok());

        let sequences = result.unwrap();
        assert!(sequences.len(1, 0) > 0);
    }

    #[tokio::test]
    async fn test_ingest_multiple_extracts() {
        let feature_pipelines = vec![
            Extract::new("PM2.5").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            Extract::new("TEMP").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
            Extract::new("WSPM").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
        ];
        let target_pipelines = vec![
            Extract::new("PM10").with_interpolation(Interpolation::Linear(LinearInterpolator::new(1))),
        ];

        let result = ingest(feature_pipelines, target_pipelines).await;
        assert!(result.is_ok());
    }
}
