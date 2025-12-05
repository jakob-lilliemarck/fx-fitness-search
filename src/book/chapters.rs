use askama::Template;

#[derive(Template)]
#[template(path = "chapters/chapter_1.html")]
pub struct Chapter1 {
    pub heading: String,
    pub base_url: String,
    pub chart_html: String,
}
