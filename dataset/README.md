# Billiards Trainer Dataset

This directory contains the dataset structure for training YOLO models to detect billiard balls and cues.

## Directory Structure

```
dataset/
├── images/          # Image files for training, validation, and testing
│   ├── train/      # Training images (80% of dataset)
│   ├── val/        # Validation images (15% of dataset)
│   └── test/       # Test images (5% of dataset)
├── labels/          # YOLO format label files
│   ├── train/      # Training labels (80% of dataset)
│   ├── val/        # Validation labels (15% of dataset)
│   └── test/       # Test labels (5% of dataset)
├── raw/            # Raw, unlabeled images for future annotation
├── README.md       # This file
└── .gitignore      # Git ignore rules
```

## Purpose of Each Directory

### images/
Contains all image files organized by dataset split. Each image should have a corresponding label file in the `labels/` directory with the same filename (but .txt extension).

- **train/**: Images used to train the model (80% of total dataset)
- **val/**: Images used to validate model performance during training (15% of total dataset)
- **test/**: Images held out for final model evaluation (5% of total dataset)

### labels/
Contains YOLO format annotation files. Each .txt file corresponds to an image file with the same name.

YOLO format per line: `<class_id> <x_center> <y_center> <width> <height>`
- All coordinates are normalized to [0, 1] relative to image dimensions
- `x_center`, `y_center`: Center point of bounding box
- `width`, `height`: Width and height of bounding box

Example class IDs:
- 0: Cue ball
- 1: Solid ball (1-7)
- 2: Striped ball (9-15)
- 3: Eight ball
- 4: Cue stick

### raw/
Contains raw, unlabeled images that have been captured but not yet annotated. These are candidates for future labeling and inclusion in the training dataset.

## Expected File Formats

### Images
- Format: JPG, PNG
- Recommended resolution: 640x640 or higher
- Color space: RGB

### Labels
- Format: .txt (YOLO format)
- Encoding: UTF-8
- One bounding box per line

## Dataset Split Ratios

The dataset should be split as follows:
- **Training**: 80% - Used to train the model
- **Validation**: 15% - Used to tune hyperparameters and monitor training
- **Test**: 5% - Used for final evaluation only

## How to Use This Dataset

### 1. Adding New Images
1. Place raw images in the `raw/` directory
2. Annotate images using a tool like LabelImg or Roboflow
3. Split annotated data into train/val/test sets according to the ratios above
4. Place images in corresponding `images/` subdirectories
5. Place label files in corresponding `labels/` subdirectories

### 2. Training with YOLO
When configuring YOLO training, set the dataset paths as follows:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
```

### 3. Maintaining Balance
Ensure that:
- Each split has a representative sample of all classes
- Lighting conditions vary across all splits
- Table angles and perspectives are diverse
- Ball arrangements include various game states

## Notes

- Image and label filenames must match exactly (except for extension)
- Keep backups of your annotated data
- Document any annotation guidelines or special cases
- Version control is configured to ignore image/label files but preserve directory structure
