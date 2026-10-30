fn main() {
    let source = image::open("thoughts/calibration/table-grid-20261026-114241.jpg")
        .unwrap()
        .to_rgb8();
    for (name, x, y) in [
        ("top-left-pocket", 39, 43),
        ("bottom-right-pocket", 1126, 564),
    ] {
        let crop = image::imageops::crop_imm(&source, x, y, 128, 128).to_image();
        crop.save(format!(
            "perception/tests/fixtures/ball-localization/{name}.png"
        ))
        .unwrap();
    }
}
