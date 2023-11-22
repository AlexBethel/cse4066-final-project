//! Image type and related functions.

use std::io::Write;

/// A grayscale, 8-bit depth image.
pub struct Image {
    xsize: usize,
    ysize: usize,

    /// Must be of length `xsize * ysize`.
    data: Vec<u8>,
}

impl Image {
    /// Create a new solid black image.
    pub fn black(xsize: usize, ysize: usize) -> Self {
        Self {
            xsize,
            ysize,
            data: std::iter::repeat(0).take(xsize * ysize).collect(),
        }
    }

    /// Convert an image to a PNG.
    pub fn write_png(&self, output: &mut impl Write) -> std::io::Result<()> {
        let mut encoder = png::Encoder::new(output, self.xsize as u32, self.ysize as u32);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.write_header()?.write_image_data(&self.data)?;
        Ok(())
    }

    /// Create an image from a function that samples pixels.
    pub fn from_sampler(xsize: usize, ysize: usize, sampler: impl Fn(usize, usize) -> u8) -> Self {
        Self {
            xsize,
            ysize,
            data: (0..xsize * ysize)
                .map(|i| sampler(i % xsize, i / xsize))
                .collect(),
        }
    }

    /// Create an image from a function that samples pixels, based on
    /// their coordinates from the center within the unit box: (0,0)
    /// is the center, (1,1) is the upper-right corner of the image,
    /// and (-1,-1) is the lower-left corner.
    pub fn from_graph_sampler(
        xsize: usize,
        ysize: usize,
        sampler: impl Fn(f64, f64) -> f64,
    ) -> Self {
        Self::from_sampler(xsize, ysize, |px, py| {
            let x_from_left = px as f64 / xsize as f64;
            let y_from_top = py as f64 / ysize as f64;
            let x = (x_from_left - 0.5) * 2.0;
            let y = -(y_from_top - 0.5) * 2.0;
            (sampler(x, y).clamp(0.0, 1.0) * 255.0) as u8
        })
    }
}
