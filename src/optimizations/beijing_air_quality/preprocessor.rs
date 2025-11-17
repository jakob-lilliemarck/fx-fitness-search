use std::f32::consts::PI;
use std::{collections::VecDeque, fmt::Display};

#[derive(Debug, Clone)]
pub struct Roc {
    offset: usize,
    buffer: VecDeque<f32>,
}

impl Roc {
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            buffer: VecDeque::with_capacity(offset + 1),
        }
    }

    fn process(&mut self, value: f32) -> Option<f32> {
        self.buffer.push_back(value);

        // Keep only the values we need
        if self.buffer.len() > self.offset + 1 {
            self.buffer.pop_front();
        }

        // Need at least offset+1 values to compute ROC
        if self.buffer.len() > self.offset {
            Some((value - self.buffer[0]) / self.offset as f32)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sin {
    period: f32,
}

impl Sin {
    pub fn new(period: f32) -> Self {
        Self { period }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        Some((2.0 * PI * value / self.period).sin())
    }
}

#[derive(Debug, Clone)]
pub struct Cos {
    period: f32,
}

impl Cos {
    pub fn new(period: f32) -> Self {
        Self { period }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        Some((2.0 * PI * value / self.period).cos())
    }
}

#[derive(Debug, Clone)]
pub struct Ema {
    /// The window size of this ema
    pub(crate) window: usize,
    /// Ema alpha, the "smoothing" factor
    pub(crate) alpha: f32,
    /// The current value
    pub(crate) value: f32,
    /// Count to track window size saturation
    pub(crate) count: usize,
}

impl Ema {
    fn new(window: usize, alpha: f32) -> Self {
        Self {
            window,
            alpha,
            value: 0.0,
            count: 0,
        }
    }

    fn process(&mut self, value: f32) -> Option<f32> {
        if self.count == 0 {
            // Initialize with first value
            self.value = value;
        } else {
            // Apply EMA formula: EMA = value * multiplier + previous_ema * (1 - multiplier)
            self.value = value * self.alpha + self.value * (1.0 - self.alpha);
        }

        self.count += 1;

        // Return None until window is saturated
        if self.count < self.window {
            None
        } else {
            Some(self.value)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Std {
    window: usize,
    buffer: VecDeque<f32>,
}

impl Std {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: window_size,
            buffer: VecDeque::with_capacity(window_size),
        }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        self.buffer.push_back(value);

        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }

        if self.buffer.len() == self.window {
            let mean: f32 = self.buffer.iter().sum::<f32>() / self.window as f32;
            let variance: f32 =
                self.buffer.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.window as f32;
            Some(variance.sqrt())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZScore {
    pub(crate) window: usize,
    pub(crate) values: VecDeque<f32>, // Ring buffer of recent values
    pub(crate) count: usize,
}

impl ZScore {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            values: VecDeque::with_capacity(window),
            count: 0,
        }
    }

    fn process(&mut self, value: f32) -> Option<f32> {
        self.values.push_back(value);
        if self.values.len() > self.window {
            self.values.pop_front();
        }
        self.count += 1;

        // If the window is not saturated, return None
        if self.count < self.window {
            return None;
        }

        // Calculate the SMA
        let mean = self.values.iter().sum::<f32>() / self.window as f32;

        // Calculate variance
        let variance =
            self.values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / self.values.len() as f32;

        // Calculate the standard deviation
        let std = variance.sqrt();

        // Handle division by zero: if std=0, all values are identical
        // In this case, zscore should be 0 (no deviation from mean)
        if std == 0.0 || std.is_nan() {
            Some(0.0)
        } else {
            Some((value - mean) / std)
        }
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Roc(Roc),
    Cos(Cos),
    Sin(Sin),
    Std(Std),
    Ema(Ema),
    ZScore(ZScore),
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Roc(roc) => write!(f, "ROC({})", roc.offset),
            Self::Cos(cos) => write!(f, "COS({})", cos.period),
            Self::Sin(sin) => write!(f, "SIN({})", sin.period),
            Self::Std(std) => write!(f, "STD({})", std.window),
            Self::Ema(ema) => write!(f, "EMA({}, {})", ema.window, ema.alpha),
            Self::ZScore(zscore) => write!(f, "ZSCORE({})", zscore.window),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unknown node: {0}")]
    UnknownNode(String),
    #[error("Invalid value: {0}")]
    InvalidValue(String),
}

impl TryFrom<&str> for Node {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let value = value.trim();

        let mut parts = value.split('(');
        let node_type = parts
            .next()
            .ok_or_else(|| Error::InvalidValue(format!("Invalid format: {}", value)))?
            .trim()
            .to_uppercase();

        let params = parts
            .next()
            .ok_or_else(|| Error::InvalidValue(format!("Missing '(' in: {}", value)))?
            .trim_end_matches(')')
            .trim();

        match node_type.as_str() {
            "ROC" => {
                let offset = params
                    .parse::<usize>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid offset, got: {}", params)))?;

                if offset < 1 {
                    return Err(Error::InvalidValue(format!(
                        "ROC offset must be at least 1, got: {}",
                        params
                    )));
                }

                Ok(Node::Roc(Roc::new(offset)))
            }
            "COS" => {
                let period = params
                    .parse::<f32>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid period, got: {}", params)))?;

                if period <= 0.0 {
                    return Err(Error::InvalidValue(format!(
                        "COS period must be larger than 0.0, got: {}",
                        params
                    )));
                }

                Ok(Node::Cos(Cos::new(period)))
            }
            "SIN" => {
                let period = params
                    .parse::<f32>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid period, got: {}", params)))?;

                if period <= 0.0 {
                    return Err(Error::InvalidValue(format!(
                        "SIN period must be larger than 0.0, got: {}",
                        params
                    )));
                }

                Ok(Node::Sin(Sin::new(period)))
            }
            "STD" => {
                let window = params
                    .parse::<usize>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid window: {}", params)))?;

                if window < 1 {
                    return Err(Error::InvalidValue(format!(
                        "STD window must be at least 1, got: {}",
                        params
                    )));
                }

                Ok(Node::Std(Std::new(window)))
            }
            "EMA" => {
                let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

                if parts.len() != 2 {
                    return Err(Error::InvalidValue(format!(
                        "EMA requires 2 parameters (window, alpha), got: {}",
                        params
                    )));
                }

                let window = parts[0]
                    .parse::<usize>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid window: {}", parts[0])))?;

                let alpha = parts[1]
                    .parse::<f32>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid alpha: {}", parts[1])))?;

                if window < 1 {
                    return Err(Error::InvalidValue(format!(
                        "EMA window must be at least 1, got: {}",
                        params
                    )));
                }

                Ok(Node::Ema(Ema::new(window, alpha)))
            }
            "ZSCORE" => {
                let window = params
                    .parse::<usize>()
                    .map_err(|_| Error::InvalidValue(format!("Invalid window: {}", params)))?;

                if window < 1 {
                    return Err(Error::InvalidValue(format!(
                        "ZSCORE window must be at least 1, got: {}",
                        params
                    )));
                }

                Ok(Node::ZScore(ZScore::new(window)))
            }
            _ => Err(Error::UnknownNode(node_type.to_string())),
        }
    }
}

impl Node {
    pub fn process(&mut self, value: f32) -> Option<f32> {
        match self {
            Self::Roc(roc) => roc.process(value),
            Self::Cos(cos) => cos.process(value),
            Self::Sin(sin) => sin.process(value),
            Self::Std(std) => std.process(value),
            Self::Ema(ema) => ema.process(value),
            Self::ZScore(zscore) => zscore.process(value),
        }
    }
}

// A simple pipeline with two preprocessing steps
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub(crate) nodes: Vec<Node>,
}

impl Pipeline {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }

    pub fn process(&mut self, value: f32) -> Option<f32> {
        let mut value: Option<f32> = Some(value);

        for node in self.nodes.iter_mut() {
            value = value.and_then(|value| node.process(value));
        }

        value
    }
}

impl Display for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node_strs: Vec<String> = self.nodes.iter().map(|n| n.to_string()).collect();
        write!(f, "{}", node_strs.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn it_computes_ema() {
        let mut ema = Ema::new(3, 0.5);

        assert_eq!(ema.process(10.0), None); // value=10.0
        assert_eq!(ema.process(20.0), None); // value=15.0
        assert_relative_eq!(ema.process(30.0).unwrap(), 22.5); // value=22.5
        assert_relative_eq!(ema.process(40.0).unwrap(), 31.25); // value=31.25
    }

    #[test]
    fn it_computes_zscore() {
        let mut zscore = ZScore::new(3);

        assert_eq!(zscore.process(10.0), None);
        assert_eq!(zscore.process(20.0), None);

        // Window [10, 20, 30]: mean=20, std≈8.165, z=(30-20)/8.165≈1.225
        assert_relative_eq!(zscore.process(30.0).unwrap(), 1.225, epsilon = 0.01);

        // Window [20, 30, 40]: mean=30, std≈8.165, z=(40-30)/8.165≈1.225
        assert_relative_eq!(zscore.process(40.0).unwrap(), 1.225, epsilon = 0.01);
    }

    #[test]
    fn it_computes_zscore_of_ema() {
        let mut ema = Ema::new(2, 0.5);
        let mut zscore = ZScore::new(2);

        assert_eq!(ema.process(10.0), None); // ema: 10.0
        let ema1 = ema.process(20.0).unwrap(); // ema: 15.0
        let ema2 = ema.process(30.0).unwrap(); // ema: 22.5
        let ema3 = ema.process(40.0).unwrap(); // ema: 31.25

        assert_eq!(zscore.process(ema1), None); // warmup

        // Window [15.0, 22.5]: mean=18.75, std=3.75, z=(22.5-18.75)/3.75=1.0
        let z1 = zscore.process(ema2).unwrap();
        assert_relative_eq!(z1, 1.0, epsilon = 0.01);

        // Window [22.5, 31.25]: mean=26.875, std=4.375, z=(31.25-26.875)/4.375=1.0
        let z2 = zscore.process(ema3).unwrap();
        assert_relative_eq!(z2, 1.0, epsilon = 0.01);
    }
}
