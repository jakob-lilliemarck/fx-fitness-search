use clap::{Parser, Subcommand};
use const_fnv1a_hash::fnv1a_hash_str_32;
use fx_durable_ga::GenotypesFilter;
use fx_durable_ga::models::{Crossover, Distribution, FitnessGoal, Mutagen, Schedule, Selector};
use fx_durable_ga_app::config::{App, ClientConfig};
use fx_durable_ga_app::optimizations::FengConfig;
use fx_durable_ga_app::optimizations::beijing_air_quality::commands::BeijingCommand;
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
        let selector =
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
        type_name: String,

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
    /// Beijing air quality domain-specific operations
    #[command(subcommand)]
    Beijing(BeijingCommand),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename(".env.shared").ok();
    dotenv::from_filename(".env.client").ok();

    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let conf = ClientConfig::from_env()?;

    let client = App::client(conf).await?;

    let args = Args::parse();

    match args.command {
        Command::Beijing(cmd) => {
            cmd.execute(client.get_svc()).await?;
        }
        Command::RequestOptimization {
            type_name,
            fitness_goal,
            schedule,
            selector,
            mutagen,
            initial_population,
        } => {
            let type_hash: i32 = fnv1a_hash_str_32(&type_name) as i32;

            tracing::info!(
                type_name = ?type_name,
                type_hash = type_hash,
                fitness_goal = ?fitness_goal,
                schedule = ?schedule,
                selector = ?selector,
                mutagen = ?mutagen,
                initial_population = ?initial_population,
                "RequestOptimization command parsed"
            );

            client
                .get_svc()
                .new_optimization_request(
                    &type_name,
                    type_hash,
                    fitness_goal,
                    schedule,
                    selector,
                    mutagen,
                    Crossover::single_point(),
                    Distribution::latin_hypercube(initial_population),
                )
                .await?;
        }
        Command::ListGenotypes { request_id } => {
            let filter = GenotypesFilter::default()
                .with_request_id(request_id)
                .with_fitness(true)
                .with_order_fitness_asc();

            let genotypes = client.get_svc().search_genotypes(&filter, 100).await?;

            println!("\nGenotypes for request {}:\n", request_id);
            println!("{:<38} {:<15}", "Genotype ID", "Fitness");
            println!("{}", "-".repeat(55));

            for (genotype, fitness) in genotypes {
                if let Some(fitness) = fitness {
                    println!("{:<38} {:<15.6}", genotype.id(), fitness);
                }
            }
        }
        Command::GetPhenotype { genotype_id } => {
            let phenotype = client
                .get_svc()
                .get_phenotype::<FengConfig>(&genotype_id)
                .await?;
            let json = serde_json::to_value(&phenotype)?;
            println!("\nPhenotype for genotype {}:\n", genotype_id);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
    }

    Ok(())
}
