use std::fs::File;

use mkpolygon::ngon_regular;

fn main() {
    // let img = mkpolygon::Image::from_graph_sampler(100, 100, |x, y| f64::hypot(x, y));
    // let img = mkpolygon::ngon_flat(1000, 1000, &[
    //     (0.7, 0.2),
    //     (0.5, 0.500001),
    //     (-0.5, 0.5),
    //     (-0.5, -0.4),
    //     (-0.3, 0.3),
    // ]);
    let img = ngon_regular(1000, 1000, 8, 0.2);
    img.divide(8, 8)
        .write_png(&mut File::create("output.png").unwrap())
        .unwrap();
}
