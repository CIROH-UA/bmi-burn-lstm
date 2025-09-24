use burn::{
    module::Param,
    nn::{
        GateController, Initializer, Linear, LinearConfig, LinearRecord, Lstm, LstmConfig,
        LstmState,
    },
    prelude::*,
    tensor::Bytes,
};
use serde_json::Value;
use std::fs;
/// Simple LSTM matching the PyTorch Nextgen_CudaLSTM architecture
#[derive(Module, Debug)]
pub struct NextgenLstm<B: Backend> {
    pub lstm: Lstm<B>,
    pub head: Linear<B>,
}

pub fn create_with_weights<B: Backend>(
    d_input: usize,
    d_output: usize,
    bias: bool,
    initializer: Initializer,
    input_record: crate::nn::LinearRecord<B>,
    hidden_record: crate::nn::LinearRecord<B>,
) -> GateController<B> {
    let l1 = LinearConfig {
        d_input,
        d_output,
        bias,
        initializer: initializer.clone(),
    }
    .init(&input_record.weight.device())
    .load_record(input_record);
    let l2 = LinearConfig {
        d_input,
        d_output,
        bias,
        initializer,
    }
    .init(&hidden_record.weight.device())
    .load_record(hidden_record);

    GateController {
        input_transform: l1,
        hidden_transform: l2,
    }
}

pub fn vec_to_tensor(input_vec: &Vec<f32>, shape: Vec<usize>) -> TensorData {
    let bytes_vec = input_vec
        .iter()
        .flat_map(|&value| f32::to_le_bytes(value))
        .collect();
    let bytes = Bytes::from_bytes_vec(bytes_vec);
    TensorData {
        bytes,
        shape,
        dtype: burn::tensor::DType::F32,
    }
}

fn create_gate_controller<B: Backend>(
    input_weights: &Vec<f32>,
    input_biases: &Vec<f32>,
    hidden_weights: &Vec<f32>,
    hidden_biases: &Vec<f32>,
    device: &Device<B>,
    input_length: usize,
    hidden_size: usize,
) -> GateController<B> {
    let input_record = LinearRecord {
        weight: Param::from_data(
            vec_to_tensor(input_weights, vec![input_length, hidden_size]),
            device,
        ),
        bias: Some(Param::from_data(
            vec_to_tensor(input_biases, vec![hidden_size]),
            device,
        )),
    };
    let hidden_record = LinearRecord {
        weight: Param::from_data(
            vec_to_tensor(hidden_weights, vec![hidden_size, hidden_size]),
            device,
        ),
        bias: Some(Param::from_data(
            vec_to_tensor(hidden_biases, vec![hidden_size]),
            device,
        )),
    };

    create_with_weights(
        input_length,
        hidden_size,
        true,
        Initializer::Zeros,
        input_record,
        hidden_record,
    )
}

impl<B: Backend> NextgenLstm<B> {
    /// Forward pass matching PyTorch implementation
    pub fn init(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> NextgenLstm<B> {
        let lstm = LstmConfig::new(input_size, hidden_size, true)
            .with_initializer(nn::Initializer::Zeros)
            .init(device);
        let head = LinearConfig::new(hidden_size, output_size)
            .with_bias(true)
            .init(device);
        Self { lstm, head }
    }
    pub fn load_json_weights(&mut self, device: &B::Device, weight_path: &str) {
        let json_str = fs::read_to_string(weight_path).expect("Failed to read file");
        let weights: Value = serde_json::from_str(&json_str).unwrap();

        fn to_vec(value: &Value) -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        }

        let input_size = weights["input_size"].as_u64().unwrap() as usize;
        let output_size = weights["output_size"].as_u64().unwrap() as usize;
        let hidden_size = weights["hidden_size"].as_u64().unwrap() as usize;

        // Extract all weights as Vecs
        let input_gate_input_weights = to_vec(&weights["lstm.input_gate.input_transform.weight"]);
        let input_gate_input_biases = to_vec(&weights["lstm.input_gate.input_transform.bias"]);
        let input_gate_hidden_weights = to_vec(&weights["lstm.input_gate.hidden_transform.weight"]);
        let input_gate_hidden_biases = to_vec(&weights["lstm.input_gate.hidden_transform.bias"]);
        self.lstm.input_gate = create_gate_controller(
            &input_gate_input_weights,
            &input_gate_input_biases,
            &input_gate_hidden_weights,
            &input_gate_hidden_biases,
            device,
            input_size,
            hidden_size,
        );

        let forget_gate_input_weights = to_vec(&weights["lstm.forget_gate.input_transform.weight"]);
        let forget_gate_input_biases = to_vec(&weights["lstm.forget_gate.input_transform.bias"]);
        let forget_gate_hidden_weights =
            to_vec(&weights["lstm.forget_gate.hidden_transform.weight"]);
        let forget_gate_hidden_biases = to_vec(&weights["lstm.forget_gate.hidden_transform.bias"]);

        self.lstm.forget_gate = create_gate_controller(
            &forget_gate_input_weights,
            &forget_gate_input_biases,
            &forget_gate_hidden_weights,
            &forget_gate_hidden_biases,
            device,
            input_size,
            hidden_size,
        );

        let cell_gate_input_weights = to_vec(&weights["lstm.cell_gate.input_transform.weight"]);
        let cell_gate_input_biases = to_vec(&weights["lstm.cell_gate.input_transform.bias"]);
        let cell_gate_hidden_weights = to_vec(&weights["lstm.cell_gate.hidden_transform.weight"]);
        let cell_gate_hidden_biases = to_vec(&weights["lstm.cell_gate.hidden_transform.bias"]);

        self.lstm.cell_gate = create_gate_controller(
            &cell_gate_input_weights,
            &cell_gate_input_biases,
            &cell_gate_hidden_weights,
            &cell_gate_hidden_biases,
            device,
            input_size,
            hidden_size,
        );

        let output_gate_input_weights = to_vec(&weights["lstm.output_gate.input_transform.weight"]);
        let output_gate_input_biases = to_vec(&weights["lstm.output_gate.input_transform.bias"]);
        let output_gate_hidden_weights =
            to_vec(&weights["lstm.output_gate.hidden_transform.weight"]);
        let output_gate_hidden_biases = to_vec(&weights["lstm.output_gate.hidden_transform.bias"]);

        self.lstm.output_gate = create_gate_controller(
            &output_gate_input_weights,
            &output_gate_input_biases,
            &output_gate_hidden_weights,
            &output_gate_hidden_biases,
            device,
            input_size,
            hidden_size,
        );

        // let head_weights = to_vec(&weights["head.weight"]);
        // let head_biases = to_vec(&weights["head.bias"]);

        // // Print to verify
        // println!("Input gate input weights: {:?}", input_gate_input_weights);
        // println!("Input gate input biases: {:?}", input_gate_input_biases);
        // println!("Input gate hidden weights: {:?}", input_gate_hidden_weights);
        // println!("Input gate hidden biases: {:?}", input_gate_hidden_biases);
        // println!("Forget gate input weights: {:?}", forget_gate_input_weights);
        // println!("Forget gate input biases: {:?}", forget_gate_input_biases);
        // println!(
        //     "Forget gate hidden weights: {:?}",
        //     forget_gate_hidden_weights
        // );
        // println!("Forget gate hidden biases: {:?}", forget_gate_hidden_biases);
        // println!("Cell gate input weights: {:?}", cell_gate_input_weights);
        // println!("Cell gate input biases: {:?}", cell_gate_input_biases);
        // println!("Cell gate hidden weights: {:?}", cell_gate_hidden_weights);
        // println!("Cell gate hidden biases: {:?}", cell_gate_hidden_biases);
        // println!("Output gate input weights: {:?}", output_gate_input_weights);
        // println!("Output gate input biases: {:?}", output_gate_input_biases);
        // println!(
        //     "Output gate hidden weights: {:?}",
        //     output_gate_hidden_weights
        // );
        // println!("Output gate hidden biases: {:?}", output_gate_hidden_biases);

        // // Repeat for other gates...
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, LstmState<B, 2>) {
        // dbg!(&input);
        let (output, state) = self.lstm.forward(input, state);
        // Apply head to each timestep
        let [batch_size, seq_length, hidden_size] = output.dims();
        // dbg!(&output);
        let output_reshaped = output.reshape([batch_size * seq_length, hidden_size]);
        let prediction = self.head.forward(output_reshaped);
        // dbg!(&prediction);
        let prediction = prediction.reshape([batch_size, seq_length, 1]);
        (prediction, state)
    }
}
