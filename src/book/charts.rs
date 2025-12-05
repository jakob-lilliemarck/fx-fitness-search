use askama::Template;
use serde::Serialize;
use uuid::Uuid;

#[derive(Template, Serialize)]
#[template(path = "charts/line.html")]
#[serde(tag = "type", content = "data", rename_all = "camelCase")]
pub enum Chart {
    Line {
        #[serde(skip)]
        id: Uuid,
        #[serde(flatten)]
        data: LineChartData,
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
pub struct ChartDataset {
    pub label: String,
    pub data: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tension: Option<f32>,
}
