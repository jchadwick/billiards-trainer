use billiards_perception::capture_settings_wire::build_capture_settings_v3;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
    let r = build_capture_settings_v3(&c).unwrap();
    std::fs::write(&a[2], serde_json::to_vec_pretty(&r).unwrap()).unwrap();
}
