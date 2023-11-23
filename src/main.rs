use std::fs::File;

use cnn::apply_kernel;
use polygon::ngon_regular;

mod cnn;
mod image;
mod polygon;

fn main() {
    // let img = mkpolygon::Image::from_graph_sampler(100, 100, |x, y| f64::hypot(x, y));
    // let img = mkpolygon::ngon_flat(1000, 1000, &[
    //     (0.7, 0.2),
    //     (0.5, 0.500001),
    //     (-0.5, 0.5),
    //     (-0.5, -0.4),
    //     (-0.3, 0.3),
    // ]);
    let img = ngon_regular(1000, 1000, 8, 0.2).divide(8, 8);

    let img = apply_kernel(&[0., 0., 0., -0.6, 0., 1., 0., 0., 0.], 3, 2, &img);
    img.write_png(&mut File::create("output.png").unwrap())
        .unwrap();
}
