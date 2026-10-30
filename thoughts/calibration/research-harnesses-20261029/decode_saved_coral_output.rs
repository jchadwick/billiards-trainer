use billiards_perception::inference::{
    decode_yolo_v8_output_with_coordinate_scale, prepare_rgb_image,
};
fn main() {
    let image = image::open("thoughts/calibration/table-grid-20261026-114241.jpg")
        .unwrap()
        .to_rgb8();
    let geometry = prepare_rgb_image(&image, 1280).unwrap().geometry;
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let ds = decode_yolo_v8_output_with_coordinate_scale(
        &[1, 6, 33600],
        &values,
        &["cue".into(), "ball".into()],
        geometry,
        0.3,
        0.45,
        16,
        args[3].parse().unwrap(),
    )
    .unwrap();
    std::fs::write(&args[2], serde_json::to_vec_pretty(&ds).unwrap()).unwrap();
}
