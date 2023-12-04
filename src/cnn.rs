//! Convolutional neural network implementation.

use std::{
    iter::{once, repeat_with},
    ops::{Index, IndexMut},
};

use rand::random;

use crate::image::Image;

pub trait NeuralNetwork {
    // Type of the input data, which is the same as the input derivatives.
    type InputElem;

    // Type of the input data, which is the same as the output derivatives.
    type OutputElem;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem>;
    fn backprop_self(
        &mut self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
        eta: f64,
    );
    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem>;

    fn stack<O: NeuralNetwork<InputElem = Self::OutputElem>>(
        self,
        other: O,
    ) -> StackedNetwork<Self, O>
    where
        Self: Sized,
    {
        StackedNetwork(self, other)
    }
}

/// A stack of network layers.
#[derive(Debug)]
pub struct StackedNetwork<A: NeuralNetwork, B: NeuralNetwork>(A, B);

impl<A: NeuralNetwork, B: NeuralNetwork<InputElem = A::OutputElem>> NeuralNetwork
    for StackedNetwork<A, B>
{
    type InputElem = A::InputElem;
    type OutputElem = B::OutputElem;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        let StackedNetwork(a, b) = self;
        b.propagate(&a.propagate(input))
    }

    fn backprop_self(
        &mut self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
        eta: f64,
    ) {
        let StackedNetwork(a, b) = self;
        let intermediates = a.propagate(inputs);
        b.backprop_self(output_derivatives, &intermediates, eta);
        let intermediate_derivatives = b.backprop_input(output_derivatives, &intermediates);
        a.backprop_self(&intermediate_derivatives, inputs, eta);
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        let StackedNetwork(a, b) = self;
        let intermediates = a.propagate(inputs);
        let intermediate_derivatives = b.backprop_input(output_derivatives, &intermediates);
        a.backprop_input(&intermediate_derivatives, inputs)
    }
}

/// A convolutional layer in a neural network. It takes as input a set
/// of images, and produces a bigger set of images (specifically, one
/// whose size is multiplied by `kernels`.length()).
#[derive(Debug)]
pub struct CnnLayer {
    kernel_diameter: usize,
    kernel_stride: usize,

    // The kernels; each one is a `kernel_diameter` x
    // `kernel_diameter` square.
    kernels: Vec<Vec<f64>>,
}

impl NeuralNetwork for CnnLayer {
    type InputElem = Image;

    type OutputElem = Image;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        input
            .iter()
            .flat_map(|input| {
                self.kernels.iter().map(|kernel| {
                    apply_kernel(kernel, self.kernel_diameter, self.kernel_stride, input)
                })
            })
            .collect()
    }

    fn backprop_self(
        &mut self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
        eta: f64,
    ) {
        // Well, first I guess let's split up the outputs into the
        // kernels that they came from.
        todo!()
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        todo!()
    }
}

pub fn apply_kernel(kernel: &[f64], diameter: usize, stride: usize, input: &Image) -> Image {
    Image::from_sampler(
        (input.xsize - diameter) / stride,
        (input.ysize - diameter) / stride,
        |x, y| {
            let ul_pixel = (x * stride, y * stride);
            let mut sum = 0.;
            for ky in 0..diameter {
                for kx in 0..diameter {
                    sum += kernel[ky * diameter + kx] * input[(ul_pixel.0 + kx, ul_pixel.1 + ky)];
                }
            }
            sum
        },
    )
}

pub struct SoftmaxLayer;

impl NeuralNetwork for SoftmaxLayer {
    type InputElem = f64;
    type OutputElem = f64;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        let sum: f64 = input.iter().map(|&x| x.exp()).sum();
        input.iter().map(|&x| x.exp() / sum).collect()
    }

    fn backprop_self(
        &mut self,
        _output_derivatives: &[Self::OutputElem],
        _inputs: &[Self::InputElem],
        _eta: f64,
    ) {
        // do nothing.
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
                -(t_c.exp() * t.exp()) / s.powi(2)
            } else {
                t_c.exp() * (s - t_c.exp()) / s.powi(2)
            }
        });

        let derr_din_i = dout_c_din_i.map(|x| x * derr_dout_c);
        derr_din_i.collect()
    }
}

/// An activation function.
pub trait Activation {
    fn apply(&self, input: f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

#[derive(Debug)]
pub struct Relu;
impl Activation for Relu {
    fn apply(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    fn derivative(&self, input: f64) -> f64 {
        input.signum().max(0.0)
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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
#[derive(Debug)]
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
