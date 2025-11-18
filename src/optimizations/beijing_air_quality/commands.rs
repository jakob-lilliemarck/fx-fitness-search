use clap::Subcommand;
use const_fnv1a_hash::fnv1a_hash_str_32;
use fx_durable_ga::models::{
    Crossover, Distribution, Encodeable, FitnessGoal, Mutagen, Schedule, Selector,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::optimizations::beijing_air_quality::evaluator::RequestVariables;

use super::phenotype::BeijingPhenotype;

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

fn parse_mutagen(s: &str) -> Result<Mutagen, String> {
    let s = s.trim();

    if let Some(inner) = s.strip_prefix("MUTAGEN(").and_then(|s| s.strip_suffix(")")) {
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "MUTAGEN expects 2 parameters (temperature, mutation_rate), got {}",
                parts.len()
            ));
        }

        let temperature: f64 = parts[0]
            .parse()
            .map_err(|_| format!("Invalid temperature value: {}", parts[0]))?;
        let mutation_rate: f64 = parts[1]
            .parse()
            .map_err(|_| format!("Invalid mutation_rate value: {}", parts[1]))?;

        Mutagen::constant(temperature, mutation_rate)
            .map_err(|e| format!("Invalid mutagen parameters: {}", e))
    } else {
        Err(format!(
            "Invalid mutagen format. Expected MUTAGEN(temperature, mutation_rate), got: {}",
            s
        ))
    }
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

        /// Mutagen: MUTAGEN(temperature, mutation_rate)
        #[arg(long, required = true, value_parser = parse_mutagen)]
        mutagen: Mutagen,

        /// Initial population size using latin hypercube
        #[arg(long, required = true)]
        initial_population: u32,

        /// Prediction horizon to predict values at
        #[arg(long, required = true)]
        prediction_horizon: usize,

        /// Target to predict including any preprocessing
        #[arg(long, required = true)]
        targets: String,
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
                mutagen,
                initial_population,
                prediction_horizon,
                targets,
            } => {
                // Hardcode type_name from BeijingPhenotype::NAME
                let type_name = BeijingPhenotype::NAME;
                let type_hash: i32 = fnv1a_hash_str_32(type_name) as i32;

                svc.new_optimization_request(
                    type_name,
                    type_hash,
                    fitness_goal,
                    schedule,
                    selector,
                    mutagen,
                    Crossover::single_point(),
                    Distribution::latin_hypercube(initial_population),
                    Some(RequestVariables::new(prediction_horizon, targets)),
                )
                .await?;

                tracing::info!("Started optimization request for Beijing air quality");
                Ok(())
            }
        }
    }
}
