use std::iter::repeat_with;

use serde::{Serialize, Deserialize};

use crate::{random::random, network::NeuralNetwork, image::Image, network::cartesian_product};

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
