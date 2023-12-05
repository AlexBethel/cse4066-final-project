use std::iter::repeat_with;

use serde::{Serialize, Deserialize};

use crate::{network::NeuralNetwork, image::Image};

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
