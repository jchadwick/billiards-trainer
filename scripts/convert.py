from ultralytics.data.converter import convert_coco

# Path to your VisioFirm-exported COCO JSON file
coco_json = "datasets/coco/annotations/coco_video.json"

# Directory where converted YOLO files will be saved
save_dir = "datasets/yolo_converted"

# Perform the conversion
convert_coco(labels_dir=coco_json, save_dir=save_dir)
