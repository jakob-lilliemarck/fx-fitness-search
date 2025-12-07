use askama::Template;
use serde::Serialize;
use uuid::Uuid;

#[derive(Template, Serialize)]
#[template(path = "charts/line.html")]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Chart {
    Line {
        #[serde(skip)]
        id: Uuid,
        data: LineChartData,
        options: Option<ChartOptions>,
    },
}

impl Chart {
    pub fn config_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn id(&self) -> &Uuid {
        match self {
            Chart::Line { id, .. } => id,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LineChartData {
    pub labels: Vec<String>,
    pub datasets: Vec<ChartDataset>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChartOptions {
    pub animation: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<f32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChartDataset {
    pub label: String,
    pub data: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tension: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_radius: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_width: Option<f32>,
}

impl ChartDataset {
    pub fn trend(&self) -> ChartDataset {
        let n = self.data.len() as f32;

        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;

        for (i, &y) in self.data.iter().enumerate() {
            let x = i as f32;
            sum_y += y;
            sum_xy += x * y;
        }

        let x_mean = (n - 1.0) / 2.0;
        let y_mean = sum_y / n;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;

        let slope = (sum_xy - n * x_mean * y_mean) / (sum_x2 - n * x_mean * x_mean);
        let intercept = y_mean - slope * x_mean;

        let trend_data = (0..self.data.len())
            .map(|i| slope * i as f32 + intercept)
            .collect();

        ChartDataset {
            label: format!("Trend: {}", self.label),
            data: trend_data,
            border_color: Some("rgb(255, 99, 132)".into()),
            tension: Some(0.0),
            point_radius: Some(0.0),
            border_width: Some(1.0),
        }
    }
}
