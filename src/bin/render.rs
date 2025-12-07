use std::path::PathBuf;

use askama::Template;
use csv::ReaderBuilder;
use fx_durable_ga_app::book::chapters::Chapter1;
use fx_durable_ga_app::book::charts::{Chart, ChartDataset, ChartOptions, LineChartData};
use uuid::Uuid;

const AUTHOR: &str = "Jakob Lilliemarck";

fn load_csv(csv_path: &PathBuf) -> Result<LineChartData, Box<dyn std::error::Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(csv_path)?;

    let mut labels = Vec::new();
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;

        let fitness = record
            .get(0)
            .and_then(|s| s.parse::<f32>().ok())
            .ok_or("Invalid fitness value")?;

        let generation = record
            .get(1)
            .and_then(|s| s.parse::<i32>().ok())
            .ok_or("Invalid generation ID")?;

        labels.push(format!("Gen {}", generation));
        data.push(fitness);
    }

    let fitness_dataset = ChartDataset {
        label: "Average Fitness".into(),
        data,
        border_color: Some("rgb(75, 192, 192)".into()),
        tension: Some(0.1),
        point_radius: Some(0.0),
        border_width: Some(1.5),
    };

    let trend_dataset = fitness_dataset.trend();

    Ok(LineChartData {
        labels,
        datasets: vec![fitness_dataset, trend_dataset],
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let csv_path = manifest_dir
        .join("data")
        .join("average_fitness_by_generation_2025-12-06T15_09_17.586384524Z.csv");

    // Load chart data from CSV
    let chart_data = load_csv(&csv_path)?;
    let chart = Chart::Line {
        id: Uuid::now_v7(),
        data: chart_data,
        options: Some(ChartOptions {
            animation: false,
            aspect_ratio: Some(1.7),
        }),
    };

    // Render chart to HTML
    let chart_html = chart.render()?;

    // Create chapter with rendered chart
    let chapter = Chapter1 {
        author: AUTHOR.to_string(),
        date: chrono::NaiveDate::from_ymd_opt(2025, 12, 6).expect("Could not create date"),
        heading: "In search of predictive ability".to_string(),
        base_url: "https://example.com".to_string(),
        chart_html,
    };

    // Render and save
    let book_dir = manifest_dir.join("book");
    let rendered_html = chapter.render()?;
    let output_path = book_dir.join("chapter-1.html");
    std::fs::write(&output_path, rendered_html)?;

    println!("Generated: {}", output_path.display());

    Ok(())
}
