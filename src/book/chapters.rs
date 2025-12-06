use askama::Template;
use chrono::NaiveDate;

#[derive(Template)]
#[template(path = "chapters/chapter_1.html")]
pub struct Chapter1 {
    pub heading: String,
    pub author: String,
    pub date: NaiveDate,
    pub base_url: String,
    pub chart_html: String,
}

impl Chapter1 {
    pub fn publication_date(&self) -> String {
        self.date.format("%B %Y").to_string()
    }
}
