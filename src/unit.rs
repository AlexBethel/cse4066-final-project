//! Unit testing utilities.

use serde::{Serialize, Deserialize};

use crate::network::NeuralNetwork;

/// A neural network layer that behaves normally, but never trains.
#[derive(Debug, Serialize, Deserialize)]
pub struct NonTrainingLayer<T>(pub T);

impl<T: NeuralNetwork> NeuralNetwork for NonTrainingLayer<T> {
    type InputElem = T::InputElem;
    type OutputElem = T::OutputElem;

    fn propagate(&self, input: &[Self::InputElem]) -> Vec<Self::OutputElem> {
        self.0.propagate(input)
    }

    fn backprop_self(
        &mut self,
        _output_derivatives: &[Self::OutputElem],
        _inputs: &[Self::InputElem],
        _eta: f64,
    ) {
        // Explicitly do nothing.
    }

    fn backprop_input(
        &self,
        output_derivatives: &[Self::OutputElem],
        inputs: &[Self::InputElem],
    ) -> Vec<Self::InputElem> {
        self.0.backprop_input(output_derivatives, inputs)
    }
}
