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

/// Floating-point, signed version of an image.
pub struct ImageDifferential {
    xsize: usize,
    ysize: usize,
    data: Vec<f64>,
}

impl From<&Image> for ImageDifferential {
    fn from(value: &Image) -> Self {
        Self {
            xsize: value.xsize,
            ysize: value.ysize,
            data: value.data.iter().map(|&x| (x as f64) / 255.0).collect(),
        }
    }
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
    type InputElem = ImageDifferential;

    type OutputElem = ImageDifferential;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        input
            .iter()
            .flat_map(|input| self.kernels.iter().map(|kernel| todo!()))
            .collect()
    }

    fn backprop(&mut self, output_derivatives: &[Self::OutputElem]) -> Vec<Self::InputElem> {
        todo!()
    }
}

fn apply_kernel(
    kernel: &[f64],
    diameter: usize,
    stride: usize,
    input: &ImageDifferential,
) -> ImageDifferential {
    
}
