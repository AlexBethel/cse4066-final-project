//! Convolutional neural network implementation.

use crate::image::Image;

pub trait NetworkLayer {
    // Type of the input data, which is the same as the input derivatives.
    type InputElem;

    // Type of the input data, which is the same as the output derivatives.
    type OutputElem;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem>;
    fn backprop(&mut self, output_derivatives: &[Self::OutputElem]) -> Vec<Self::InputElem>;
}

/// A convolutional layer in a neural network. It takes as input a set
/// of images, and produces a bigger set of images (specifically, one
/// whose size is multiplied by `kernels`.length()).
pub struct CnnLayer {
    n_inputs: usize,

    kernel_diameter: usize,
    kernel_stride: usize,

    // The kernels; each one is a `kernel_diameter` x
    // `kernel_diameter` square.
    kernels: Vec<Vec<f64>>,
}

impl NetworkLayer for CnnLayer {
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

    fn backprop(&mut self, output_derivatives: &[Self::OutputElem]) -> Vec<Self::InputElem> {
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
