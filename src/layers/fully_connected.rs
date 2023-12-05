use std::{
    iter::{once, repeat_with},
    ops::{Index, IndexMut},
};

use serde::{Deserialize, Serialize};

use crate::{network::NeuralNetwork, random::random};

/// An activation function.
pub trait Activation {
    fn apply(&self, input: f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Linear;
impl Activation for Linear {
    fn apply(&self, input: f64) -> f64 {
        input
    }

    fn derivative(&self, _: f64) -> f64 {
        1.
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Relu;
impl Activation for Relu {
    fn apply(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    fn derivative(&self, input: f64) -> f64 {
        input.signum().max(0.0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LeakyRelu(pub f64);
impl Activation for LeakyRelu {
    fn apply(&self, input: f64) -> f64 {
        if input > 0. {
            input
        } else {
            input * self.0
        }
    }

    fn derivative(&self, input: f64) -> f64 {
        if input > 0. {
            1.
        } else {
            self.0
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Logistic;
impl Activation for Logistic {
    fn apply(&self, input: f64) -> f64 {
        1. / (1. + input.exp())
    }

    fn derivative(&self, input: f64) -> f64 {
        -input.exp() / (1. + input.exp()).powi(2)
    }
}

/// A standard neural network layer.
#[derive(Debug, Serialize, Deserialize)]
pub struct FullyConnectedLayer<F: Activation> {
    n_inputs: usize,
    n_outputs: usize,

    // n_outputs x (n_inputs + 1) matrix; the extra 1 is for the bias.
    weights: Vec<f64>,

    activation: F,
}

impl<F: Activation> Index<(usize, usize)> for FullyConnectedLayer<F> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (output_number, input_number) = index;
        &self.weights[output_number + input_number * self.n_outputs]
    }
}

impl<F: Activation> IndexMut<(usize, usize)> for FullyConnectedLayer<F> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (output_number, input_number) = index;
        &mut self.weights[output_number + input_number * self.n_outputs]
    }
}

impl<F: Activation> FullyConnectedLayer<F> {
    pub fn new(n_inputs: usize, n_outputs: usize, activation: F) -> Self {
        Self {
            n_inputs,
            n_outputs,
            weights: repeat_with(|| random::<f64>() - 0.5)
                .take((n_inputs + 1) * n_outputs)
                .collect(),
            activation,
        }
    }
}

impl<F: Activation> NeuralNetwork for FullyConnectedLayer<F> {
    type InputElem = f64;

    type OutputElem = f64;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        (0..self.n_outputs)
            .map(|out_idx| {
                self.activation.apply(
                    self.activation
                        .apply(self.output_neuron_summation(out_idx, input)),
                )
            })
            .collect()
    }

    fn backprop_self(
        &mut self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
        eta: f64,
    ) {
        for (output_number, &derror_doutput) in output_derivatives.iter().enumerate() {
            // (Bias is irrelevant here.)
            let summation = self.output_neuron_summation(output_number, inputs);
            let a_prime = self.activation.derivative(summation);
            if a_prime == 0.0 {
                // Optimization: if the neuron is turned off, then
                // we're guaranteed no changes ever.
                continue;
            }

            for (input_number, input) in (inputs.iter().cloned().chain(once(1.0))).enumerate() {
                let doutput_dweight = a_prime * input;
                let derror_dweight = derror_doutput * doutput_dweight;
                self[(output_number, input_number)] -= eta * derror_dweight;
            }
        }
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        (0..self.n_inputs)
            .map(|input_idx| {
                output_derivatives
                    .iter()
                    .enumerate()
                    .map(|(output_idx, derror_doutput)| {
                        let sum = self.output_neuron_summation(output_idx, inputs);
                        let doutput_dsum = self.activation.derivative(sum);
                        let dsum_dinput = self[(output_idx, input_idx)];
                        let doutput_dinput = doutput_dsum * dsum_dinput;
                        let derror_dinput = derror_doutput * doutput_dinput;
                        derror_dinput
                    })
                    .sum()
            })
            .collect()
    }
}

impl<F: Activation> FullyConnectedLayer<F> {
    fn output_neuron_summation(&self, output_number: usize, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .cloned()
            // Extra 1 is for the bias.
            .chain(once(1.0))
            .enumerate()
            .map(|(in_idx, in_value)| self[(output_number, in_idx)] * in_value)
            .sum()
    }
}
