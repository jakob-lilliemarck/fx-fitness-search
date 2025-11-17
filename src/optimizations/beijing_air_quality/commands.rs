use super::dataset::VALID_COLUMNS;
use super::model::{FeedForward, SequenceModel};
use super::preprocessor::{Node, Pipeline};
use burn::backend::Autodiff;
use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::data::dataloader::Dataset;
use clap::{Parser, Subcommand};
use serde::Serialize;
use tracing::{Level, info};

const PATHS: &[&str] = &[
    "data/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "data/PRSA_Data_Changping_20130301-20170228.csv",
    "data/PRSA_Data_Dingling_20130301-20170228.csv",
    "data/PRSA_Data_Dongsi_20130301-20170228.csv",
    "data/PRSA_Data_Guanyuan_20130301-20170228.csv",
    "data/PRSA_Data_Gucheng_20130301-20170228.csv",
    "data/PRSA_Data_Huairou_20130301-20170228.csv",
    "data/PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "data/PRSA_Data_Shunyi_20130301-20170228.csv",
    "data/PRSA_Data_Tiantan_20130301-20170228.csv",
    "data/PRSA_Data_Wanliu_20130301-20170228.csv",
    "data/PRSA_Data_Wanshouxigong_20130301-20170228.csv", // saved for inference!
];

const WANSHOUXIGONG_PATH: &str = "data/PRSA_Data_Wanshouxigong_20130301-20170228.csv";

#[derive(Serialize)]
struct ResultOutput {
    validation_loss: f64,
}

/// Parse a single key-value pair with optional pipeline
/// Format: output_name=source_column:TRANSFORM1 TRANSFORM2 ...
/// Or:     output_name=source_column  (no preprocessing)
fn parse_feature_pipeline(
    s: &str,
) -> Result<(String, String, Pipeline), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    let output_name = s[..pos].to_string();
    let rest = &s[pos + 1..];

    // Check if preprocessing is specified with colon
    let (source_column, pipeline_str) = if let Some(colon_pos) = rest.find(':') {
        (rest[..colon_pos].to_string(), &rest[colon_pos + 1..])
    } else {
        // If no colon, no preprocessing - just use the source column directly
        (rest.to_string(), "")
    };

    // Validate source column name
    if !VALID_COLUMNS.contains(&source_column.as_str()) {
        return Err(format!(
            "Invalid source column: '{}'. Valid columns are: {}",
            source_column,
            VALID_COLUMNS.join(", ")
        )
        .into());
    }

    // Parse pipeline (left-to-right execution order)
    let nodes: Vec<Node> = if pipeline_str.trim().is_empty() {
        vec![]
    } else {
        pipeline_str
            .split_whitespace()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| Node::try_from(s))
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok((output_name, source_column, Pipeline::new(nodes)))
}

#[derive(Debug, Parser)]
#[command(
    name = "example-rnn-beijing-air-quality",
    version = "1.0",
    author = "Jakob",
    about = "Trains a neural network on Beijing air quality data"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train the model
    Train {
        // ============================================================
        // Model params
        // ============================================================
        /// Hidden layer size (hyperparameter to tune)
        #[arg(long, required = true)]
        hidden_size: usize,

        /// Learning rate
        #[arg(long, required = true)]
        learning_rate: f64,

        // ============================================================
        // Dataset & Preprocessing params
        // ============================================================
        /// Feature pipelines as key=value pairs (e.g., temp_ema=TEMP:ZSCORE(100) or hour_sin=hour:SIN(24))
        #[clap(long = "feature", value_parser = parse_feature_pipeline)]
        features: Vec<(String, String, Pipeline)>,

        /// Target pipelines as key=value pairs (e.g., target_temp=TEMP or target_pm25=PM2.5:ZSCORE(100))
        #[clap(long = "target", value_parser = parse_feature_pipeline)]
        targets: Vec<(String, String, Pipeline)>,

        // ============================================================
        // Training params
        // ============================================================
        /// Sequence length
        #[arg(long, required = true)]
        sequence_length: usize,

        /// Prediction horizon
        #[arg(long, required = true)]
        prediction_horizon: usize,

        /// Batch size
        #[clap(long, default_value_t = 8)]
        batch_size: usize,

        /// Epochs
        #[clap(long, default_value_t = 100)]
        epochs: usize,

        /// Path to save the trained model (optional)
        #[arg(long)]
        model_save_path: Option<String>,
    },

    /// Export preprocessed dataset to CSV
    Export {
        /// Feature pipelines as key=value pairs (e.g., PM2.5=PM2.5:ROC(1),ZSCORE(100) or hour_sin=hour:SIN(24))
        #[clap(long = "feature", value_parser = parse_feature_pipeline)]
        features: Vec<(String, String, Pipeline)>,

        /// Output CSV file path
        #[arg(long, required = true)]
        output: String,
    },

    /// Run inference on Wanshouxigong dataset using a trained model
    Infer {
        /// Path to saved model (will load from {model_path}.config.json)
        #[arg(long, required = true)]
        model_path: String,

        /// Maximum number of predictions to generate (optional, defaults to all rows)
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_feature_with_preprocessing() {
        let result = parse_feature_pipeline("output=TEMP:ZSCORE(10)");
        assert!(result.is_ok());
        let (output_name, source_column, pipeline) = result.unwrap();
        assert_eq!(output_name, "output");
        assert_eq!(source_column, "TEMP");
        assert_eq!(pipeline.nodes.len(), 1);
    }

    #[test]
    fn test_parse_feature_without_preprocessing() {
        let result = parse_feature_pipeline("target_temp=TEMP");
        assert!(result.is_ok());
        let (output_name, source_column, pipeline) = result.unwrap();
        assert_eq!(output_name, "target_temp");
        assert_eq!(source_column, "TEMP");
        assert_eq!(pipeline.nodes.len(), 0); // No preprocessing
    }

    #[test]
    fn test_parse_feature_invalid_column() {
        let result = parse_feature_pipeline("output=INVALID_COL");
        assert!(result.is_err());
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .pretty()
        .init();

    type Backend = Autodiff<NdArray>;
    let device = NdArrayDevice::default();

    let args = Args::parse();

    match args.command {
        Command::Train {
            hidden_size,
            learning_rate,
            features,
            targets,
            sequence_length,
            prediction_horizon,
            batch_size,
            epochs,
            model_save_path,
        } => {
            let feature_length = features.len();
            let target_length = targets.len();

            // Load data from all weather stations and combine their sequences
            let mut all_train_items = Vec::new();
            let mut all_valid_items = Vec::new();

            for (i, path) in PATHS.iter().enumerate() {
                info!(
                    message = format!("Loading file [{}/{}]", i + 1, PATHS.len()),
                    station = path,
                    path
                );

                let dataset_builder =
                    super::dataset::build_dataset_from_file(path, &features, &targets)?;
                let (train, opt_valid) =
                    dataset_builder.build(sequence_length, prediction_horizon, Some(0.8))?;
                let valid = opt_valid.expect("must get a validation dataset during training");

                let train_len = train.len();

                // Collect items and metadata from this station
                for idx in 0..train_len {
                    if let Some(metadata) = train.get_metadata(idx) {
                        if let Some(item) = train.get(idx) {
                            all_train_items.push((metadata.clone(), item));
                        }
                    }
                }
                for idx in 0..valid.len() {
                    if let Some(metadata) = valid.get_metadata(idx) {
                        if let Some(item) = valid.get(idx) {
                            all_valid_items.push((metadata.clone(), item));
                        }
                    }
                }
            }

            info!(
                message = "Loading and preprocessing completed",
                training_sequences = all_train_items.len(),
                validation_sequences = all_valid_items.len()
            );

            // Create combined datasets
            let dataset_training = super::dataset::SequenceDataset::from_items(all_train_items);
            let dataset_validation = super::dataset::SequenceDataset::from_items(all_valid_items);

            // Choose model architecture:
            let model = FeedForward::<Backend>::new(
                &device,
                feature_length,
                hidden_size,
                target_length,
                sequence_length,
            );
            // let model = SimpleRnn::<Backend>::new(&device, feature_length, hidden_size, target_length, sequence_length);
            // let model = SimpleLstm::<Backend>::new(&device, feature_length, hidden_size, target_length, sequence_length);

            // Build config for saving
            let train_config = if model_save_path.is_some() {
                let feature_defs: Vec<String> = features
                    .iter()
                    .map(|(name, col, pipeline)| {
                        if pipeline.nodes.is_empty() {
                            format!("{}={}", name, col)
                        } else {
                            format!("{}={}:{}", name, col, pipeline)
                        }
                    })
                    .collect();

                let target_defs: Vec<String> = targets
                    .iter()
                    .map(|(name, col, pipeline)| {
                        if pipeline.nodes.is_empty() {
                            format!("{}={}", name, col)
                        } else {
                            format!("{}={}:{}", name, col, pipeline)
                        }
                    })
                    .collect();

                Some(super::train_config::TrainConfig {
                    hidden_size,
                    input_size: feature_length,
                    output_size: target_length,
                    sequence_length,
                    prediction_horizon,
                    features: feature_defs,
                    targets: target_defs,
                })
            } else {
                None
            };

            // Train
            let (_model, validation_loss) = super::train::train(
                &device,
                &dataset_training,
                &dataset_validation,
                epochs,
                batch_size,
                learning_rate,
                model,
                model_save_path,
                train_config,
            );

            // Output JSON result to stdout for GA to consume
            let result = ResultOutput {
                validation_loss: validation_loss as f64,
            };
            println!("{}", serde_json::to_string(&result).unwrap());
        }

        Command::Export { features, output } => {
            // For export, use features as both features and targets from first station
            let dataset_builder =
                super::dataset::build_dataset_from_file(PATHS[0], &features, &features)?;
            dataset_builder.to_csv(&output)?;
            tracing::info!("Preprocessed dataset exported to: {}", output);
        }

        Command::Infer { model_path, limit } => {
            let engine = super::inference::InferenceEngine::<Backend>::load(&model_path, &device)?;
            println!("Running inference...");
            let results = super::infer_dataset::run_inference(&engine, WANSHOUXIGONG_PATH, limit)?;

            // Write results to CSV
            let csv_path = format!("{}.infer.csv", model_path);
            super::infer_dataset::write_results_to_csv(&results, &csv_path)?;
            println!("Results written to: {}", csv_path);

            // Calculate and display metrics
            let metrics = super::infer_dataset::calculate_metrics(&results);
            println!("\n=== Inference Summary ===");
            println!("{}", metrics);
        }
    }

    Ok(())
}
