use std::iter::repeat;

use serde::{Deserialize, Serialize};

use crate::{network::NeuralNetwork, network::cartesian_product, polygon::Image};

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
