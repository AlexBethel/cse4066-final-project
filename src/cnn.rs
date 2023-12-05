//! Convolutional neural network implementation.

use std::{
    iter::{once, repeat, repeat_with},
    ops::{Index, IndexMut},
};

use serde::{Deserialize, Serialize};

// use rand::random;
use crate::random::random;

use crate::image::Image;

pub trait NeuralNetwork {
    // Type of the input data, which is the same as the input derivatives.
    type InputElem;

    // Type of the input data, which is the same as the output derivatives.
    type OutputElem;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem>;
    fn backprop_self(
        &mut self,
        _output_derivatives: &[Self::OutputElem],
        _inputs: &[Self::InputElem],
        _eta: f64,
    ) {
        // do nothing by default.
    }
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
#[derive(Debug, Serialize, Deserialize)]
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
#[derive(Debug, Serialize, Deserialize)]
pub struct CnnLayer {
    kernel_diameter: usize,
    kernel_stride: usize,

    // The kernels; each one is a `kernel_diameter` x
    // `kernel_diameter` square.
    kernels: Vec<Vec<f64>>,
}

impl CnnLayer {
    pub fn new(kernel_diameter: usize, kernel_stride: usize, n_kernels: usize) -> Self {
        Self {
            kernel_diameter,
            kernel_stride,
            kernels: repeat_with(|| {
                repeat_with(|| random::<f64>() - 0.5)
                    .take(kernel_diameter * kernel_diameter)
                    .collect()
            })
            .take(n_kernels)
            .collect(),
        }
    }
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

    fn backprop_self(&mut self, output_derivatives: &[Image], inputs: &[Image], eta: f64) {
        for (_index, (kernel, dl_dout)) in self
            .kernels
            .iter_mut()
            .zip(output_derivatives.iter())
            .enumerate()
        {
            for (x, y) in cartesian_product(0..self.kernel_diameter, 0..self.kernel_diameter) {
                // if index == 3 && x == 1 && y == 0 {
                    let mut dl_dfilterxy = 0.;
                    for (i, j) in cartesian_product(0..dl_dout.xsize, 0..dl_dout.ysize) {
                        let dl_doutij = dl_dout[(i, j)];
                        let doutij_dfilterxy = inputs[0][(x + i, y + j)];
                        dl_dfilterxy += dl_doutij * doutij_dfilterxy;
                    }

                    // println!("Analytic dl_dfilterxy = {dl_dfilterxy}");
                    // println!("adjusting by {}", -dl_dfilterxy * eta);
                    // println!(
                    //     "expecting delta loss {}",
                    //     -dl_dfilterxy * eta * dl_dfilterxy
                    // );
                    kernel[y * self.kernel_diameter + x] -= dl_dfilterxy * eta;
                // }
            }
        }
    }

    fn backprop_input(
        &self,
        _output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        println!("cnn backprop");

        // unimplemented
        inputs
            .iter()
            .map(|_| Image::black(inputs[0].xsize, inputs[0].ysize))
            .collect()
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
        println!("softmax backprop");

        // Index of the single correct output choice.
        let (c, derr_dout_c) = output_derivatives
            .iter()
            .enumerate()
            .find(|(_, &v)| v != 0.)
            .expect("network was already perfect!");
        let t_c = not_nan(inputs[c]);
        println!("got t_c");

        let s: f64 = not_nan(inputs.iter().map(|&t| not_nan(t.exp())).sum());
        println!("got s = {s}");
        let dout_c_din_i = inputs.iter().enumerate().map(|(i, &t)| {
            // dbg!(i);
            // if i == 4 {
            //     dbg!(t);
            //     dbg!(t_c);
            //     dbg!(s);
            // }
            let x = not_nan(if i != c {
                // -(t_c.exp() * t.exp()) / s.powi(2)
                -(t_c + t - 2. * s.ln()).exp()
            } else {
                println!("snd");
                // dbg!(t_c.exp());
                // dbg!(s - t_c.exp());
                // dbg!(t_c.exp() * (s - t_c.exp()));
                // dbg!(s.powi(2));
                println!("{}", (t_c + (s - t_c.exp()).ln() - 2. * s.ln()).exp());
                // if (t_c.exp() * (s - t_c.exp()) / s.powi(2)).is_nan() {
                //     panic!();
                // }
                (t_c + (s - t_c.exp()).ln() - 2. * s.ln()).exp()
                // t_c.exp() * (s - t_c.exp()) / s.powi(2)
            });
            dbg!(x);
            x
        });
        println!("made dout_c_din_i");

        dbg!(derr_dout_c);
        let derr_din_i = dout_c_din_i.map(|x| not_nan(x * derr_dout_c));
        println!("made derr_din_i");
        let res = derr_din_i.collect();
        println!("collected");
        res
    }
}

pub fn softmax_loss_gradient(outputs: &[f64], correct_output: usize) -> Vec<f64> {
    let mut result = vec![0.; outputs.len()];
    result[correct_output] = -(1. / outputs[correct_output]).min(1e10);
    result
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MaxpoolLayer {
    diameter: usize,
}

impl MaxpoolLayer {
    pub fn new(diameter: usize) -> Self {
        Self { diameter }
    }
}

impl NeuralNetwork for MaxpoolLayer {
    type InputElem = Image;

    type OutputElem = Image;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        input
            .iter()
            .map(|image| {
                Image::from_sampler(
                    image.xsize / self.diameter,
                    image.ysize / self.diameter,
                    |tx, ty| {
                        let sx = tx * self.diameter;
                        let sy = ty * self.diameter;
                        (sx..sx + self.diameter)
                            .flat_map(|x| repeat(x).zip(sy..sy + self.diameter))
                            .map(|(x, y)| image[(x, y)])
                            .max_by(f64::total_cmp)
                            .expect("maxpool diameter can't be empty")
                    },
                )
            })
            .collect()
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        output_derivatives
            .iter()
            .zip(inputs.iter())
            .map(|(output_derivative, input)| {
                let mut input_derivative = Image::black(input.xsize, input.ysize);
                for tx in 0..output_derivative.xsize {
                    for ty in 0..output_derivative.ysize {
                        let sx = tx * self.diameter;
                        let sy = ty * self.diameter;
                        // let max_coord = (sx..sx + self.diameter)
                        //     .flat_map(|x| repeat(x).zip(sy..sy + self.diameter))
                        let max_coord =
                            cartesian_product(sx..sx + self.diameter, sy..sy + self.diameter)
                                .max_by(|&(ax, ay), &(bx, by)| {
                                    f64::total_cmp(&input[(ax, ay)], &input[(bx, by)])
                                })
                                .expect("maxpool diameter can't be empty");
                        input_derivative[max_coord] = output_derivative[(tx, ty)];
                    }
                }
                input_derivative
            })
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FlattenLayer;

impl NeuralNetwork for FlattenLayer {
    type InputElem = Image;

    type OutputElem = f64;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        input
            .iter()
            .flat_map(|image| image.data.iter())
            .cloned()
            .collect()
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        let n_imgs = inputs.len();
        let xsize = inputs[0].xsize;
        let ysize = inputs[0].ysize;

        let mut imgs: Vec<_> = repeat_with(|| Image::black(xsize, ysize))
            .take(n_imgs)
            .collect();

        imgs.iter_mut()
            .flat_map(|img| img.data.iter_mut())
            .zip(output_derivatives.iter())
            .for_each(|(input_derivative, &output_derivative)| {
                *input_derivative = output_derivative;
            });

        imgs
    }
}

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
                self[(output_number, input_number)] -= not_nan(eta * derror_dweight);
            }
        }
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        println!("fully connected backprop");
        (0..self.n_inputs)
            .map(|input_idx| {
                not_nan(
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
                        .sum(),
                )
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

fn cartesian_product<T, U>(left: T, right: U) -> impl Iterator<Item = (T::Item, U::Item)>
where
    T: Iterator,
    U: Iterator + Clone + 'static,
    T::Item: Clone,
{
    left.flat_map(move |l| repeat(l).zip(right.clone()))
}

fn not_nan(x: f64) -> f64 {
    if x.is_nan() {
        panic!()
    } else {
        x
    }
}
