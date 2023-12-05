use std::{f64::consts::TAU, fs::File, path::Path};

use crate::{layers::recompose, random::random};
use image::Image;
use layers::{
    cnn::CnnLayer,
    flatten::FlattenLayer,
    fully_connected::{FullyConnectedLayer, Linear},
    maxpool::MaxpoolLayer,
    recompose::RecomposeLayer,
    softmax::{softmax_loss_gradient, SoftmaxLayer},
};
use mnist::read_mnist;
use network::{NeuralNetwork, StackedNetwork};
use unit::NonTrainingLayer;

mod image;
mod layers;
mod mnist;
mod network;
mod polygon;
mod random;
mod unit;

const MAX_SIDES: usize = 8;

fn main() {
    // generate_network();

    // dump_testing_data();

    test_network(load_network());

    // unittest_network();

    // train_mnist();
    test_mnist(load_mnist_network());

    // generate_ae_network();
}

type NetworkType = StackedNetwork<
    StackedNetwork<
        StackedNetwork<StackedNetwork<NonTrainingLayer<CnnLayer>, MaxpoolLayer>, FlattenLayer>,
        FullyConnectedLayer<Linear>,
    >,
    SoftmaxLayer,
>;

type MNistNetworkType = StackedNetwork<
    StackedNetwork<
        StackedNetwork<StackedNetwork<NonTrainingLayer<CnnLayer>, MaxpoolLayer>, FlattenLayer>,
        FullyConnectedLayer<Linear>,
    >,
    SoftmaxLayer,
>;

fn generate_network() {
    let mut network: NetworkType = NonTrainingLayer(CnnLayer::new(3, 1, 8))
        .stack(MaxpoolLayer::new(8))
        .stack(FlattenLayer)
        .stack(FullyConnectedLayer::new(1152, MAX_SIDES + 1, Linear))
        .stack(SoftmaxLayer);

    for _ in 0..10000 {
        let n_sides = random::<u32>() % (MAX_SIDES as u32 - 2) + 3;
        let image = polygon::ngon_regular(100, 100, n_sides, random::<f64>() * TAU);
        let inputs = [image];

        let output = network.propagate(&inputs);
        let loss = -output[n_sides as usize].ln();
        println!("{:?}", (n_sides, loss, &output));

        let output_derivatives = softmax_loss_gradient(&output, n_sides.try_into().unwrap());
        network.backprop_self(&output_derivatives, &inputs, 0.01);
    }

    bincode::serialize_into(File::create("network.dat").unwrap(), &network).unwrap();
}

fn generate_ae_network() {
    let mut network = NonTrainingLayer(CnnLayer::new(3, 1, 8))
        .stack(MaxpoolLayer::new(8))
        .stack(FlattenLayer)
        .stack(FullyConnectedLayer::new(1152, MAX_SIDES + 1, Linear))
        .stack(FullyConnectedLayer::new(MAX_SIDES + 1, 28 * 28, Linear))
        .stack(RecomposeLayer::new(28, 28));

    for _ in 0..100 {
        // let n_sides = random::<u32>() % (MAX_SIDES as u32 - 2) + 3;
        let n_sides = 7;
        // let angle = random::<f64>() * TAU;
        let angle = 3.1;
        let image = polygon::ngon_regular(100, 100, n_sides, angle);
        let inputs = [image];

        let output = network.propagate(&inputs);
        let loss = recompose::loss(&output[0], &inputs[0]);
        println!("{:?}", (n_sides, loss));

        output[0]
            .write_png(&mut File::create("test.png").unwrap())
            .unwrap();
        let output_derivatives = recompose::loss_gradient(&output[0], &inputs[0]);
        network.backprop_self(&[output_derivatives], &inputs, 0.01);
    }

    bincode::serialize_into(File::create("ae_network.dat").unwrap(), &network).unwrap();
}

fn train_mnist() {
    let mut network: MNistNetworkType = NonTrainingLayer(CnnLayer::new(4, 1, 8))
        .stack(MaxpoolLayer::new(4))
        .stack(FlattenLayer)
        .stack(FullyConnectedLayer::new(288, 10, Linear))
        .stack(SoftmaxLayer);
    let mnist = read_mnist(
        &Path::new("mnist/train-images-idx3-ubyte.gz"),
        &Path::new("mnist/train-labels-idx1-ubyte.gz"),
    );

    for (idx, item) in mnist.iter().enumerate() {
        let inputs = [item.into()];

        let output = network.propagate(&inputs);
        let loss = -output[item.category as usize].ln();
        if idx % 100 == 0 {
            println!("{idx}: {:?}", (item.category, loss, &output));
        }

        let output_derivatives = softmax_loss_gradient(&output, item.category.try_into().unwrap());
        network.backprop_self(&output_derivatives, &inputs, 0.01);
    }

    bincode::serialize_into(File::create("mnist_network.dat").unwrap(), &network).unwrap();
}

fn test_mnist(network: MNistNetworkType) {
    let mnist = read_mnist(
        &Path::new("mnist/t10k-images-idx3-ubyte.gz"),
        &Path::new("mnist/t10k-labels-idx1-ubyte.gz"),
    );

    let mut right = 0;
    let mut wrong = 0;
    for (idx, item) in mnist.iter().enumerate() {
        let image: Image = item.into();
        image
            .write_png(&mut File::create(format!("mnist_test/{idx:04}.png")).unwrap())
            .unwrap();
        let inputs = [image];

        let output = network.propagate(&inputs);
        let loss = -output[item.category as usize].ln();

        let ident = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        if ident == item.category as usize {
            right += 1;
        } else {
            wrong += 1;
        }
        println!(
            "{}: {} -> {}{}",
            idx,
            loss,
            ident,
            if ident != item.category as usize {
                " WRONG"
            } else {
                ""
            }
        );
    }

    println!("Accuracy {}%", 100 * right / (right + wrong));
}

fn load_mnist_network() -> MNistNetworkType {
    bincode::deserialize_from(
        File::open("mnist_network.dat").expect("Can't open mnist_network.dat"),
    )
    .expect("Malformed network")
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
