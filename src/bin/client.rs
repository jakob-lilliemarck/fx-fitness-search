use clap::{Parser, Subcommand};
use fx_durable_ga_app::config::ClientConfig;
use fx_durable_ga::models::{FitnessGoal, Mutagen, Schedule, Selector};
use sqlx::types::Uuid;

// ============================================================
// Clap value parsers for complex types
// ============================================================

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

        Ok(Selector::tournament(tournament_size, sample_size))
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

#[derive(Debug, Parser)]
#[command(
    name = "fx-durable-ga-app",
    version = "1.0",
    author = "Jakob",
    about = "Distributed optimizations"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Request an optimization
    RequestOptimization {
        /// The name of the evaluatbale feature configuration
        #[arg(long, required = true)]
        name: String,

        /// Valid arguments:
        /// - MIN(goal: f32)
        /// - MAX(goal: f32)
        #[arg(long, required = true, value_parser = parse_fitness_goal)]
        fitness_goal: FitnessGoal,

        /// Valid arguments:
        /// - GENERATIONAL(generations: u32, population: u32)
        /// - ROLLING(evaluations: u32, population: u32, interval: u32)
        #[arg(long, required = true, value_parser = parse_schedule)]
        schedule: Schedule,

        /// Valid arguments:
        /// - TOURNAMENT(tournament_size: usize, sample_size: usize)
        /// - ROULETTE(sample_size: usize)
        #[arg(long, required = true, value_parser = parse_selector)]
        selector: Selector,

        /// Valid arguments:
        /// - MUTAGEN(temperature: f32, mutation_rate: f32)
        #[arg(long, required = true, value_parser = parse_mutagen)]
        mutagen: Mutagen,

        /// Size of the initial population, using latin hypercube
        #[arg(long, required = true)]
        initial_population: u32,
    },
    /// List genotypes and their evaluated fitness for a request
    ListGenotypes {
        #[arg(long, required = true)]
        request_id: Uuid,
    },
    /// Decode a genotype genome to it's phenotype representation
    GetPhenotype {
        #[arg(long, required = true)]
        genotype_id: Uuid,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(false)
        .with_level(true)
        .init();

    let conf = ClientConfig::from_env()?;

    // Parse command line arguments
    let args = Args::parse();

    // Log the parsed command
    match &args.command {
        Command::RequestOptimization {
            name,
            fitness_goal,
            schedule,
            selector,
            mutagen,
            initial_population,
        } => {
            tracing::info!(
                name = ?name,
                fitness_goal = ?fitness_goal,
                schedule = ?schedule,
                selector = ?selector,
                mutagen = ?mutagen,
                initial_population = ?initial_population,
                "RequestOptimization command parsed"
            );
        }
        Command::ListGenotypes { request_id } => {
            tracing::info!(
                request_id = ?request_id,
                "ListGenotypes command parsed"
            );
        }
        Command::GetPhenotype { genotype_id } => {
            tracing::info!(
                genotype_id = ?genotype_id,
                "GetPhenotype command parsed"
            );
        }
    }

    Ok(())
}
