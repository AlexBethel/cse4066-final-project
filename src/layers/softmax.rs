use serde::{Deserialize, Serialize};

use crate::network::NeuralNetwork;

#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxLayer;

impl NeuralNetwork for SoftmaxLayer {
    type InputElem = f64;
    type OutputElem = f64;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        let sum: f64 = input.iter().map(|&x| x.exp()).sum();
        input.iter().map(|&x| x.exp() / sum).collect()
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        // Index of the single correct output choice.
        let (c, derr_dout_c) = output_derivatives
            .iter()
            .enumerate()
            .find(|(_, &v)| v != 0.)
            .expect("network was already perfect!");
        let t_c = inputs[c];

        let s: f64 = inputs.iter().map(|&t| t.exp()).sum();
        let dout_c_din_i = inputs.iter().enumerate().map(|(i, &t)| {
            if i != c {
                -(t_c + t - 2. * s.ln()).exp()
            } else {
                (t_c + (s - t_c.exp()).ln() - 2. * s.ln()).exp()
            }
        });

        let derr_din_i = dout_c_din_i.map(|x| x * derr_dout_c);
        let res = derr_din_i.collect();
        res
    }
}

pub fn softmax_loss_gradient(outputs: &[f64], correct_output: usize) -> Vec<f64> {
    let mut result = vec![0.; outputs.len()];
    result[correct_output] = -(1. / outputs[correct_output]).min(1e10);
    result
}
