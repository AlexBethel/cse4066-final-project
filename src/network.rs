//! Convolutional neural network implementation.

use std::iter::repeat;

use serde::{Deserialize, Serialize};

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
pub struct StackedNetwork<A, B>(A, B);

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

pub fn cartesian_product<T, U>(left: T, right: U) -> impl Iterator<Item = (T::Item, U::Item)>
where
    T: Iterator,
    U: Iterator + Clone + 'static,
    T::Item: Clone,
{
    left.flat_map(move |l| repeat(l).zip(right.clone()))
}
