use burn::nn;
use burn::prelude::*;

/// Trait for models that process sequence inputs
pub trait SequenceModel<B: Backend>: Module<B> + Sized {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        seq_len: usize,
    ) -> Self;

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2>;
}

#[derive(Module, Debug)]
pub struct SimpleRnn<B: Backend> {
    w_ih: nn::Linear<B>,       // input → hidden
    w_hh: nn::Linear<B>,       // hidden → hidden
    linear_out: nn::Linear<B>, // hidden → output
}

impl<B: Backend> SequenceModel<B> for SimpleRnn<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        _seq_len: usize, // Unused, for API consistency with FeedForward
    ) -> Self {
        Self {
            w_ih: nn::LinearConfig::new(input_size, hidden_size).init(device),
            w_hh: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            linear_out: nn::LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, _, _] = input.dims();
        let hidden_size = self.w_hh.weight.dims()[0];

        let mut h = Tensor::<B, 2>::zeros([batch_size, hidden_size], &input.device());

        // Split along dimension 1 (sequence dimension)
        let timesteps: Vec<_> = input.split(1, 1);

        for x_t in timesteps {
            // x_t: [batch, 1, input_size] → squeeze dim 1
            let x_t = x_t.squeeze_dim(1);
            let x_proj = self.w_ih.forward(x_t);
            let h_proj = self.w_hh.forward(h);
            h = (x_proj + h_proj).tanh();
        }

        self.linear_out.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear_out: nn::Linear<B>,
}

impl<B: Backend> SequenceModel<B> for FeedForward<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        seq_len: usize,
    ) -> Self {
        let flattened_input_size = seq_len * input_size;

        Self {
            linear1: nn::LinearConfig::new(flattened_input_size, hidden_size).init(device),
            linear2: nn::LinearConfig::new(hidden_size, hidden_size).init(device),
            linear_out: nn::LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, seq_len, features] = input.dims();

        // Flatten sequence and features into single dimension
        let flattened = input.reshape([batch_size, seq_len * features]);

        // Two hidden layers with ReLU activation
        let x = burn::tensor::activation::relu(self.linear1.forward(flattened));
        let x = burn::tensor::activation::relu(self.linear2.forward(x));

        // Output layer
        self.linear_out.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct SimpleLstm<B: Backend> {
    lstm: nn::Lstm<B>,
    linear_out: nn::Linear<B>,
}

impl<B: Backend> SequenceModel<B> for SimpleLstm<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        _seq_len: usize, // Unused, for API consistency with FeedForward
    ) -> Self {
        Self {
            lstm: nn::LstmConfig::new(input_size, hidden_size, true).init(device),
            linear_out: nn::LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Forward pass through LSTM
        let (output, _state) = self.lstm.forward(input, None);

        let seq_len = output.dims()[1];
        let last_step = output.narrow(1, seq_len - 1, 1).squeeze::<2>();

        // Linear projection
        self.linear_out.forward(last_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray;

    #[test]
    fn test_feedforward_forward_shape() {
        let device = Default::default();
        let model = FeedForward::<B>::new(&device, 4, 8, 2, 3);

        let input = Tensor::<B, 3>::zeros([2, 3, 4], &device); // [batch=2, seq=3, features=4]
        let output = model.forward(input);

        assert_eq!(output.dims(), [2, 2]); // [batch=2, output=2]
    }

    #[test]
    fn test_feedforward_different_batch_sizes() {
        let device = Default::default();
        let model = FeedForward::<B>::new(&device, 4, 8, 2, 3);

        for batch_size in [1, 4, 16] {
            let input = Tensor::<B, 3>::zeros([batch_size, 3, 4], &device);
            let output = model.forward(input);

            assert_eq!(output.dims(), [batch_size, 2]);
        }
    }

    #[test]
    fn test_feedforward_produces_output() {
        let device = Default::default();
        let model = FeedForward::<B>::new(&device, 4, 8, 2, 3);

        let input = Tensor::<B, 3>::ones([1, 3, 4], &device);
        let output = model.forward(input);

        // Output should be finite (not NaN or Inf)
        let output_data = output.clone().into_data();
        for val in output_data.bytes.chunks_exact(4) {
            let f = f32::from_le_bytes([val[0], val[1], val[2], val[3]]);
            assert!(f.is_finite(), "Output contains non-finite value: {}", f);
        }
    }

    #[test]
    #[should_panic]
    fn test_feedforward_seq_len_mismatch_panics() {
        let device = Default::default();
        // Model built for seq_len=3, input_size=4
        let model = FeedForward::<B>::new(&device, 4, 8, 2, 3);
        // Provide seq_len=4 -> flatten size 16, but layer expects 12
        let bad_input = Tensor::<B, 3>::zeros([1, 4, 4], &device);
        let _ = model.forward(bad_input);
    }

    #[test]
    fn test_feedforward_record_round_trip() {
        let device = Default::default();
        let model = FeedForward::<B>::new(&device, 4, 8, 2, 3);
        let record = model.clone().into_record();

        let model2 = FeedForward::<B>::new(&device, 4, 8, 2, 3).load_record(record);

        let input = Tensor::<B, 3>::ones([2, 3, 4], &device);
        let out1 = model.forward(input.clone()).into_data().bytes;
        let out2 = model2.forward(input).into_data().bytes;
        assert_eq!(out1, out2);
    }
}
