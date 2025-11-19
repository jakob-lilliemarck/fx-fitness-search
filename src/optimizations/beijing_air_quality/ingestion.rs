use std::collections::HashMap;

use crate::core::preprocessor::Transform;
use crate::core::dataset::DatasetBuilder;

use super::parser::read_csv;

// Valid source columns from Beijing air quality CSV files
// These are columns that can be converted to f32 for preprocessing
// CSV header: No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station
// Excluded: No (u32 identifier), year (not used), station (no f32 conversion)
// Included: wd (WindDirection converts to f32 degrees)
pub const SOURCE_COLUMNS: &[&str] = &[
    "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
    "RAIN", "wd", "WSPM",
]; // 15 columns total (indices 0-14)

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

/// Build a DatasetBuilder from a CSV file
pub fn build_dataset_from_file(
    path: &str,
    features: &[Transform],
    targets: &[Transform],
) -> anyhow::Result<DatasetBuilder> {
    // Process features (clone pipelines for each file)
    let mut feature_pipelines = HashMap::with_capacity(features.len());
    let mut feature_output_names = Vec::with_capacity(features.len());
    let mut feature_source_columns = Vec::with_capacity(features.len());

    for transform in features {
        feature_pipelines.insert(transform.destination.clone(), transform.pipeline.clone());
        feature_output_names.push(transform.destination.clone());
        feature_source_columns.push(transform.source.clone());
    }

    // Process targets (clone pipelines for each file)
    let mut target_pipelines = HashMap::with_capacity(targets.len());
    let mut target_output_names = Vec::with_capacity(targets.len());
    let mut target_source_columns = Vec::with_capacity(targets.len());

    for transform in targets {
        target_pipelines.insert(transform.destination.clone(), transform.pipeline.clone());
        target_output_names.push(transform.destination.clone());
        target_source_columns.push(transform.source.clone());
    }

    // Collect all unique source columns needed
    let mut all_source_columns = feature_source_columns.clone();
    for col in &target_source_columns {
        if !all_source_columns.contains(col) {
            all_source_columns.push(col.clone());
        }
    }

    let mut dataset_builder = DatasetBuilder::new(
        feature_pipelines,
        feature_output_names,
        feature_source_columns,
        target_pipelines,
        target_output_names,
        target_source_columns,
        Some(35066),
    );

    // Read and push all rows from specified file
    for result in read_csv(path)? {
        let row = result?;

        // Create a record with ALL the source columns needed
        let mut record = HashMap::new();
        let mut has_missing = false;

        for source_column in &all_source_columns {
            let wd: Option<f32> = row.wd.clone().map(|wd| wd.into());
            let value = match source_column.as_str() {
                "PM2.5" => row.pm2_5,
                "PM10" => row.pm10,
                "SO2" => row.so2,
                "NO2" => row.no2,
                "CO" => row.co,
                "O3" => row.o3,
                "TEMP" => row.temp,
                "PRES" => row.pres,
                "DEWP" => row.dewp,
                "RAIN" => row.rain,
                "WSPM" => row.wspm,
                "hour" => Some(row.hour as f32),
                "day" => Some(row.day as f32),
                "month" => Some(row.month as f32),
                "wd" => wd,
                _ => None,
            };

            if let Some(v) = value {
                record.insert(source_column.clone(), v);
            } else {
                has_missing = true;
            }
        }

        // Skip rows with missing values
        if has_missing {
            continue;
        }

        // Push to builder (skips rows where pipeline returns None)
        dataset_builder.push(record, row.no.to_string())?;
    }

    Ok(dataset_builder)
}
