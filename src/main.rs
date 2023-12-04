use std::fs::File;

use cnn::{apply_kernel, FullyConnectedLayer, NeuralNetwork, Relu};
use polygon::ngon_regular;

use crate::cnn::{LeakyRelu, Logistic};

mod cnn;
mod image;
mod polygon;

fn main() {
    // // let img = mkpolygon::Image::from_graph_sampler(100, 100, |x, y| f64::hypot(x, y));
    // // let img = mkpolygon::ngon_flat(1000, 1000, &[
    // //     (0.7, 0.2),
    // //     (0.5, 0.500001),
    // //     (-0.5, 0.5),
    // //     (-0.5, -0.4),
    // //     (-0.3, 0.3),
    // // ]);
    // let img = ngon_regular(1000, 1000, 8, 0.2).divide(8, 8);

    // let img = apply_kernel(&[0., 0., 0., -0.6, 0., 1., 0., 0., 0.], 3, 2, &img);
    // img.write_png(&mut File::create("output.png").unwrap())
    //     .unwrap();

    let mut layer = FullyConnectedLayer::new(1, 1, Logistic)
        // .stack(FullyConnectedLayer::new(8, 8, Relu))
        // .stack(FullyConnectedLayer::new(8, 8, Relu))
        .stack(FullyConnectedLayer::new(1, 1, Logistic));
    dbg!(&layer);

    println!("training...");
    let xes = [1.0, 1.5, 2.0, 2.5, 3.0];
    let function = |x| {
        // (if x >= 2.5 {
        //     x + 5.0
        // } else {
            x / 10.0
        // }) / 10.0
    };
    for input in xes.iter().cloned().map(|x| x * 1.0).cycle().take(100) {
        let output = layer.propagate(&[input])[0];
        // let expected = input + 5.0;
        // let expected = if input >= 2.5 {
        //     input + 5.0
        // } else {
        //     input + 4.0
        // };
        let expected = function(input);
        // dbg!(output);

        // let err = expected - input;
        // dbg!(err);

        let derror_doutput = 2.0 * (output - expected);
        layer.backprop_self(&[derror_doutput], &[input], 0.01);
    }
    dbg!(&layer);
    // dbg!((layer.propagate(&[1.0]), 1.0 + 4.0));
    // dbg!(layer.propagate(&[1.5]));
    // dbg!(layer.propagate(&[2.0]));
    // dbg!(layer.propagate(&[2.5]));
    // dbg!(layer.propagate(&[3.0]));
    for &v in xes.iter() {
        dbg!((v, layer.propagate(&[v]), function(v)));
    }
}
