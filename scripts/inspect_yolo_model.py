#!/usr/bin/env python3
"""Inspect YOLO model to see what classes it actually has."""

import onnx
import onnxruntime as ort

def inspect_model(model_path):
    """Inspect ONNX model metadata and class names."""

    print(f"Inspecting model: {model_path}\n")

    # Load ONNX model
    model = onnx.load(model_path)

    # Print metadata
    print("=== Model Metadata ===")
    for prop in model.metadata_props:
        print(f"  {prop.key}: {prop.value}")

    # Try to load with ONNX Runtime to get more info
    print("\n=== ONNX Runtime Session Info ===")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    print(f"Input names: {[i.name for i in session.get_inputs()]}")
    print(f"Input shapes: {[i.shape for i in session.get_inputs()]}")
    print(f"Output names: {[o.name for o in session.get_outputs()]}")
    print(f"Output shapes: {[o.shape for o in session.get_outputs()]}")

    # Check for class names in model metadata
    meta = session.get_modelmeta()
    print(f"\nModel metadata keys: {meta.custom_metadata_map.keys()}")
    for key, value in meta.custom_metadata_map.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    inspect_model("models/yolov8n-pool-1280.onnx")  # High-res 1280x1280 model
