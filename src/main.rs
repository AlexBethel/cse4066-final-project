use std::fs::File;

fn main() {
    let img = mkpolygon::Image::from_graph_sampler(100, 100, |x, y| x.hypot(y));
    img.write_png(&mut File::create("output.png").unwrap())
        .unwrap();
}
