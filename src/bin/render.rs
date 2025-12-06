use std::path::PathBuf;

use askama::Template;
use fx_durable_ga_app::book::chapters::Chapter1;
use fx_durable_ga_app::book::charts::{Chart, ChartDataset, LineChartData};
use uuid::Uuid;

const AUTHOR: &str = "Jakob Lilliemarck";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create chart from data (later: load from CSV)
    let chart = Chart::Line {
        id: Uuid::now_v7(),
        data: LineChartData {
            labels: vec!["Jan".into(), "Feb".into(), "Mar".into()],
            datasets: vec![ChartDataset {
                label: "Sales".into(),
                data: vec![65.0, 59.0, 80.0],
                border_color: Some("rgb(255, 0, 128)".into()),
                tension: Some(0.1),
            }],
        },
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
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let book_dir = manifest_dir.join("book");
    let rendered_html = chapter.render()?;
    let output_path = book_dir.join("chapter-1.html");
    std::fs::write(&output_path, rendered_html)?;

    println!("Generated: {}", output_path.display());

    Ok(())
}
