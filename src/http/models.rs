use serde::Deserialize;

#[derive(Deserialize)]
pub struct SaxpyInput {
    pub a: f32,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}
