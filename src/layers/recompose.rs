//! Layer that takes numbers as input, and composes them into an
//! image.

use serde::{Deserialize, Serialize};

use crate::{
    image::Image,
    network::{cartesian_product, NeuralNetwork},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct RecomposeLayer {
    xsize: usize,
    ysize: usize,
}

impl RecomposeLayer {
    pub fn new(xsize: usize, ysize: usize) -> Self {
        Self { xsize, ysize }
    }
}

impl NeuralNetwork for RecomposeLayer {
    type InputElem = f64;

    type OutputElem = Image;

    fn propagate(&self, input: &[f64]) -> Vec<Image> {
        vec![Image::from_sampler(self.xsize, self.ysize, |x, y| {
            input[x + y * self.xsize]
        })]
    }

    fn backprop_input(&self, output_derivatives: &[Image], inputs: &[f64]) -> Vec<f64> {
        cartesian_product(0..self.ysize, 0..self.xsize)
            .map(|(y, x)| output_derivatives[0][(x, y)])
            .collect()
    }
}

pub fn loss(output: &Image, expected: &Image) -> f64 {
    cartesian_product(0..output.xsize, 0..output.ysize)
        .map(|(x, y)| (output[(x, y)] - expected[(x, y)]).powi(2))
        .sum()
}

pub fn loss_gradient(output: &Image, expected: &Image) -> Image {
    Image::from_sampler(output.xsize, output.ysize, |x, y| {
        2. * (expected[(x, y)] - output[(x, y)])
    })
}
