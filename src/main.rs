use std::fs::File;

use cnn::{apply_kernel, FullyConnectedLayer, NeuralNetwork, Relu};
use polygon::ngon_regular;

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

    let mut layer = FullyConnectedLayer::new(2, 1, Relu);
    dbg!(&layer);

    println!("training...");
    let xes = [1.0, 1.5, 2.0, 2.5, 3.0];
    for input in xes.iter().cloned().cycle().take(1000) {
        let output = layer.propagate(&[input, input])[0];
        let expected = input * 0.5 + 0.25;
        // dbg!(output);

        // let err = expected - input;
        // dbg!(err);

        let derror_doutput = 2.0 * (output - expected);
        layer.backprop_self(&[derror_doutput], &[input, input], 0.1);
    }
    dbg!(&layer);
    dbg!(layer.propagate(&[1.0, 1.0]));
    dbg!(layer.propagate(&[3.0, 3.0]));
}
