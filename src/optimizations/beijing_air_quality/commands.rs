use super::BeijingGenotypeManager;
use super::protocol::RequestVariables;
use clap::Subcommand;
use fx_durable_ga::models::{FitnessGoal, Schedule, Selector};
use std::sync::Arc;

// Value parsers for clap (copied from client.rs)

fn parse_fitness_goal(s: &str) -> Result<FitnessGoal, String> {
    let s = s.trim();

    if let Some(inner) = s.strip_prefix("MIN(").and_then(|s| s.strip_suffix(")")) {
        let threshold: f64 = inner
            .trim()
            .parse()
            .map_err(|_| format!("Invalid threshold value in MIN({})", inner))?;
        FitnessGoal::minimize(threshold).map_err(|e| format!("Invalid fitness goal: {}", e))
    } else if let Some(inner) = s.strip_prefix("MAX(").and_then(|s| s.strip_suffix(")")) {
        let threshold: f64 = inner
            .trim()
            .parse()
            .map_err(|_| format!("Invalid threshold value in MAX({})", inner))?;
        FitnessGoal::maximize(threshold).map_err(|e| format!("Invalid fitness goal: {}", e))
    } else {
        Err(format!(
            "Invalid fitness goal format. Expected MIN(threshold) or MAX(threshold), got: {}",
            s
        ))
    }
}

fn parse_schedule(s: &str) -> Result<Schedule, String> {
    let s = s.trim();

    if let Some(inner) = s
        .strip_prefix("GENERATIONAL(")
        .and_then(|s| s.strip_suffix(")"))
    {
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "GENERATIONAL expects 2 parameters (generations, population), got {}",
                parts.len()
            ));
        }

        let generations: u32 = parts[0]
            .parse()
            .map_err(|_| format!("Invalid generations value: {}", parts[0]))?;
        let population: u32 = parts[1]
            .parse()
            .map_err(|_| format!("Invalid population value: {}", parts[1]))?;

        Ok(Schedule::generational(generations, population))
    } else if let Some(inner) = s.strip_prefix("ROLLING(").and_then(|s| s.strip_suffix(")")) {
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() != 3 {
            return Err(format!(
                "ROLLING expects 3 parameters (evaluations, population, interval), got {}",
                parts.len()
            ));
        }

        let evaluations: u32 = parts[0]
            .parse()
            .map_err(|_| format!("Invalid evaluations value: {}", parts[0]))?;
        let population: u32 = parts[1]
            .parse()
            .map_err(|_| format!("Invalid population value: {}", parts[1]))?;
        let interval: u32 = parts[2]
            .parse()
            .map_err(|_| format!("Invalid interval value: {}", parts[2]))?;

        Ok(Schedule::rolling(evaluations, population, interval))
    } else {
        Err(format!(
            "Invalid schedule format. Expected GENERATIONAL(generations, population) or ROLLING(evaluations, population, interval), got: {}",
            s
        ))
    }
}

fn parse_selector(s: &str) -> Result<Selector, String> {
    let s = s.trim();

    if let Some(inner) = s
        .strip_prefix("TOURNAMENT(")
        .and_then(|s| s.strip_suffix(")"))
    {
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "TOURNAMENT expects 2 parameters (tournament_size, sample_size), got {}",
                parts.len()
            ));
        }

        let tournament_size: usize = parts[0]
            .parse()
            .map_err(|_| format!("Invalid tournament_size value: {}", parts[0]))?;
        let sample_size: usize = parts[1]
            .parse()
            .map_err(|_| format!("Invalid sample_size value: {}", parts[1]))?;
        let selector: Selector =
            Selector::tournament(tournament_size, sample_size).map_err(|err| err.to_string())?;
        Ok(selector)
    } else if let Some(inner) = s
        .strip_prefix("ROULETTE(")
        .and_then(|s| s.strip_suffix(")"))
    {
        let sample_size: usize = inner
            .trim()
            .parse()
            .map_err(|_| format!("Invalid sample_size value: {}", inner))?;

        Ok(Selector::roulette(sample_size))
    } else {
        Err(format!(
            "Invalid selector format. Expected TOURNAMENT(tournament_size, sample_size) or ROULETTE(sample_size), got: {}",
            s
        ))
    }
}

fn parse_mutation_rate(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|_| format!("Could not parse mutation_rate as f64: {}", s))?;

    if value > 1.0 || value < 0.0 {
        return Err(format!(
            "mutation rate must be a number between 0.0 and 1.0. Received {}",
            value
        ));
    }

    return Ok(value);
}

fn parse_temperature(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|_| format!("Could not parse temperature as f64: {}", s))?;

    if value > 1.0 || value < 0.0 {
        return Err(format!(
            "Temperature must be a number between 0.0 and 1.0. Received {}",
            value
        ));
    }

    return Ok(value);
}

#[derive(Debug, Subcommand)]
pub enum BeijingCommand {
    /// Request a genetic algorithm optimization for Beijing air quality prediction
    RequestOptimization {
        /// Fitness goal: MIN(threshold) or MAX(threshold)
        #[arg(long, required = true, value_parser = parse_fitness_goal)]
        fitness_goal: FitnessGoal,

        /// Schedule: GENERATIONAL(generations, population) or ROLLING(evaluations, population, interval)
        #[arg(long, required = true, value_parser = parse_schedule)]
        schedule: Schedule,

        /// Selector: TOURNAMENT(tournament_size, sample_size) or ROULETTE(sample_size)
        #[arg(long, required = true, value_parser = parse_selector)]
        selector: Selector,

        /// The likeliness of any one gene mutating
        #[arg(long, required = true, value_parser = parse_mutation_rate)]
        mutation_rate: f64,

        /// The magnitude of mutation changes
        #[arg(long, required=true, value_parser = parse_temperature)]
        temperature: f64,

        /// Prediction horizon to predict values at
        #[arg(long, required = true)]
        prediction_horizon: usize,

        /// Number of epochs to train for
        #[arg(long, default_value = "25")]
        epochs: usize,

        /// Early stopping patience: number of epochs without improvement before stopping
        #[arg(long, default_value = "5")]
        patience: usize,

        /// Epoch at which to start validation
        #[arg(long, default_value = "10")]
        validation_start_epoch: usize,

        /// Batch size for training and validation
        #[arg(long, default_value = "100")]
        batch_size: usize,
    },
}

impl BeijingCommand {
    pub async fn execute(
        self,
        svc: Arc<fx_durable_ga::optimization::Service>,
    ) -> anyhow::Result<()> {
        match self {
            Self::RequestOptimization {
                fitness_goal,
                schedule,
                selector,
                mutation_rate,
                temperature,
                prediction_horizon,
                epochs,
                patience,
                validation_start_epoch,
                batch_size,
            } => {
                // Hardcode type_name from BeijingGenotypeManager::TYPE_NAME
                let type_name = BeijingGenotypeManager::TYPE_NAME;
                let type_hash: i32 = BeijingGenotypeManager::TYPE_HASH;

                let training = RequestVariables::with_training_params(
                    prediction_horizon,
                    epochs,
                    patience,
                    validation_start_epoch,
                    batch_size,
                );

                let user_defined = serde_json::json!({
                    "mutation_rate": mutation_rate,
                    "temperature": temperature,
                    "training": training
                });

                svc.new_optimization_request(
                    type_name, // FIXME - really the optimization request should only need one of type_name and type_hash, no need for both?
                    type_hash,
                    fitness_goal,
                    schedule,
                    selector,
                    user_defined,
                    Option::<()>::None,
                )
                .await?;

                tracing::info!("Started optimization request for Beijing air quality");
                Ok(())
            }
        }
    }
}
