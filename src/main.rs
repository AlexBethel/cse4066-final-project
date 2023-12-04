use std::{f64::consts::TAU, fs::File};

use cnn::{CnnLayer, MaxpoolLayer, NeuralNetwork};
use image::Image;

use crate::{
    cnn::{softmax_loss_gradient, FlattenLayer, FullyConnectedLayer, Linear, SoftmaxLayer},
    random::random,
};

mod cnn;
mod image;
mod polygon;
mod random;

const MAX_SIDES: usize = 12;

fn main() {
    generate_network();

    dump_testing_data();

    // 82 is the threshold
    test_network(load_network());
}

type NetworkType = cnn::StackedNetwork<
    cnn::StackedNetwork<
        cnn::StackedNetwork<cnn::StackedNetwork<CnnLayer, MaxpoolLayer>, FlattenLayer>,
        FullyConnectedLayer<Linear>,
    >,
    SoftmaxLayer,
>;

fn generate_network() {
    let mut network: NetworkType = CnnLayer::new(3, 1, 8)
        .stack(MaxpoolLayer::new(8))
        .stack(FlattenLayer)
        .stack(FullyConnectedLayer::new(1152, MAX_SIDES + 1, Linear))
        .stack(SoftmaxLayer);

    for _ in 0..10000 {
        let n_sides = random::<u32>() % (MAX_SIDES as u32 - 2) + 3;
        let image = polygon::ngon_regular(100, 100, n_sides, random::<f64>() * TAU);
        // image
        //     .write_png(&mut File::create(format!("img_{:05}.png", i)).unwrap())
        //     .unwrap();
        let inputs = [image];

        let output = network.propagate(&inputs);
        let loss = -output[n_sides as usize].ln();
        println!("{:?}", (n_sides, loss, &output));

        let output_derivatives = softmax_loss_gradient(&output, n_sides.try_into().unwrap());
        network.backprop_self(&output_derivatives, &inputs, 0.01);
    }

    bincode::serialize_into(File::create("network.dat").unwrap(), &network).unwrap();
}

fn load_network() -> NetworkType {
    bincode::deserialize_from(File::open("network.dat").expect("Can't open network.dat"))
        .expect("Malformed network")
}

fn test_network(network: NetworkType) {
    let mut right = 0;
    let mut wrong = 0;
    for n_sides in 3..=MAX_SIDES {
        for index in 0..100 {
            let filename = format!("testing/img_{n_sides}_{index:02}.png");
            let image = Image::read_png(&mut File::open(&filename).expect("Failed to open file"))
                .expect("Malformed image");
            let output = network.propagate(&[image]);
            let loss = -output[n_sides as usize].ln();

            let ident = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            if ident == n_sides {
                right += 1;
            } else {
                wrong += 1;
            }
            println!(
                "{}: {} -> {}{}",
                filename,
                loss,
                ident,
                if ident != n_sides { " WRONG" } else { "" }
            );
        }
    }

    println!("Accuracy {}%", 100 * right / (right + wrong));
}

fn dump_testing_data() {
    for n_sides in 3..=MAX_SIDES {
        let side_angle = TAU / (n_sides as f64);
        for index in 0..100 {
            println!("{:?}", (n_sides, index));
            let angle = (index as f64) * side_angle / 100.0;
            polygon::ngon_regular(100, 100, n_sides as _, angle)
                .write_png(
                    &mut File::create(format!("testing/img_{n_sides}_{index:02}.png"))
                        .expect("Failed to create file"),
                )
                .expect("Failed to write file");
        }
    }
}
