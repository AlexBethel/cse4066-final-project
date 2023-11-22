//! Library to generate images of polygons.

mod image;

use std::f64::consts::TAU;

pub use image::Image;

/// Draw an N-polygon, in white with a black background with no
/// antialiasing, centered within the image.
pub fn ngon_flat(xsize: usize, ysize: usize, vertices: &[(f64, f64)]) -> Image {
    Image::from_graph_sampler(xsize, ysize, |x, y| {
        let mut multiplicity = 0;
        for (a, b) in Edges::new(vertices) {
            let (a, b) = if a.0 > b.0 {
                (b, a)
            } else {
                (a, b)
            };

            if a.0 <= x && b.0 > x {
                let ab = (b.0 - a.0, b.1 - a.1);
                let nab = (-ab.1, ab.0);
                let ap = (x - a.0, y - a.1);

                let nab_ap = nab.0 * ap.0 + nab.1 * ap.1;
                multiplicity += (nab_ap > 0.0) as u32;
            }
        }

        (multiplicity % 2 == 1) as u32 as f64
    })
}

/// Draw a regular n-gon in the center of an image.
pub fn ngon_regular(xsize: usize, ysize: usize, n_vertices: u32) -> Image {
    let vertices: Vec<_> = (0..n_vertices)
        .map(|i| {
            let angle = i as f64 * TAU / n_vertices as f64;
            let sc = angle.sin_cos();
            (sc.1, sc.0)
        })
        .collect();
    ngon_flat(xsize, ysize, &vertices)
}

/// An iterator around the edges of a polygon.
struct Edges<'a> {
    vertices: &'a [(f64, f64)],
    index: usize,
}

impl<'a> Edges<'a> {
    pub fn new(vertices: &'a [(f64, f64)]) -> Self {
        Self { vertices, index: 0 }
    }
}

impl Iterator for Edges<'_> {
    type Item = ((f64, f64), (f64, f64));

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.vertices.len() {
            None
        } else {
            let res = Some((
                self.vertices[self.index],
                self.vertices[(self.index + 1) % self.vertices.len()],
            ));
            self.index += 1;
            res
        }
    }
}
