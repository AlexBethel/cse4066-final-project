//! Loader for MNIST data.

use std::{fs::File, io::Read, path::Path};

use flate2::read::GzDecoder;

use crate::image::Image;

pub struct MNistDigit {
    pub category: u32,
    pub data: [u8; 28 * 28],
}

impl From<&MNistDigit> for Image {
    fn from(value: &MNistDigit) -> Self {
        Self::from_sampler(28, 28, |x, y| value.data[y * 28 + x] as f64 / 255.0)
    }
}

pub fn read_mnist(images_filename: &Path, labels_filename: &Path) -> Vec<MNistDigit> {
    let imgs = read_images(images_filename);
    let labels = read_labels(labels_filename);

    imgs.into_iter()
        .zip(labels.into_iter())
        .map(|(data, category)| MNistDigit { category, data })
        .collect()
}

fn read_images(filename: &Path) -> Vec<[u8; 28 * 28]> {
    let mut stream = GzDecoder::new(File::open(filename).unwrap());
    let mut data = Vec::new();
    stream.read_to_end(&mut data).unwrap();
    drop(stream);
    let data = data;

    let (magic, data) = data.split_at(4);
    assert_eq!(magic, [0, 0, 8, 3]);

    let (num_images, data) = data.split_at(4);
    let _num_images = u32::from_be_bytes(*<&[u8; 4]>::try_from(num_images).unwrap());

    let (width, data) = data.split_at(4);
    let width = u32::from_be_bytes(*<&[u8; 4]>::try_from(width).unwrap());
    let (height, data) = data.split_at(4);
    let height = u32::from_be_bytes(*<&[u8; 4]>::try_from(height).unwrap());
    assert_eq!(width, 28);
    assert_eq!(height, 28);

    // And now we read the individual pixels.
    data.chunks_exact((width * height) as usize)
        .map(|chunk| chunk.try_into().unwrap())
        .collect()
}

fn read_labels(filename: &Path) -> Vec<u32> {
    let mut stream = GzDecoder::new(File::open(filename).unwrap());
    let mut data = Vec::new();
    stream.read_to_end(&mut data).unwrap();
    drop(stream);
    let data = data;

    let (magic, data) = data.split_at(4);
    assert_eq!(magic, [0, 0, 8, 1]);

    let (num_labels, data) = data.split_at(4);
    let num_labels = u32::from_be_bytes(*<&[u8; 4]>::try_from(num_labels).unwrap());

    data.iter()
        .map(|&byte| byte as _)
        .take(num_labels as _)
        .collect()
}
