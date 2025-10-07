#!/usr/bin/env python3
"""
Download Billiard Ball Datasets from Roboflow

This script downloads billiard ball detection datasets from Roboflow Universe
and optionally merges them into a single training dataset.

Usage:
    # Set your API key first
    export ROBOFLOW_API_KEY="your_key_here"

    # Download datasets
    python tools/download_roboflow_datasets.py

    # Download and merge into single dataset
    python tools/download_roboflow_datasets.py --merge
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List
import yaml

try:
    from roboflow import Roboflow
except ImportError:
    print("❌ ERROR: roboflow package not installed!")
    print("\nPlease install it:")
    print("  pip install roboflow")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent


def get_api_key() -> str:
    """Get Roboflow API key from environment or prompt user"""
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        print("\n🔑 Roboflow API Key Required")
        print("="*70)
        print("\nTo get your API key:")
        print("  1. Sign up/login at https://roboflow.com")
        print("  2. Go to Account Settings (click your profile)")
        print("  3. Copy your Private API Key")
        print("\nYou can set it as an environment variable:")
        print("  export ROBOFLOW_API_KEY='your_key_here'")
        print("\nOr enter it now (will not be saved):")

        api_key = input("\nAPI Key: ").strip()

        if not api_key:
            print("❌ No API key provided")
            sys.exit(1)

    return api_key


def download_dataset(rf: Roboflow, workspace: str, project: str, version: int,
                     output_dir: Path, dataset_name: str) -> Path:
    """Download a single dataset from Roboflow"""
    print(f"\n📥 Downloading {dataset_name}...")
    print(f"   Workspace: {workspace}")
    print(f"   Project: {project}")
    print(f"   Version: {version}")

    try:
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(
            "yolov8",
            location=str(output_dir / dataset_name)
        )

        dataset_path = Path(dataset.location)
        print(f"✅ Downloaded to: {dataset_path}")

        return dataset_path

    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e}")
        raise


def merge_datasets(dataset_paths: List[Path], output_dir: Path) -> Path:
    """Merge multiple YOLO datasets into one"""
    print("\n🔀 Merging datasets...")

    merged_dir = output_dir / "merged_billiards_dataset"
    merged_dir.mkdir(exist_ok=True)

    # Create directory structure
    for split in ['train', 'valid', 'test']:
        (merged_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (merged_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect all class names
    all_classes = set()

    # Copy files from each dataset
    file_counter = 0
    for dataset_path in dataset_paths:
        print(f"  Processing {dataset_path.name}...")

        # Read data.yaml to get class names
        yaml_path = dataset_path / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    all_classes.update(data['names'])

        # Copy image and label files
        for split in ['train', 'valid', 'test']:
            # Roboflow datasets have structure: train/images/, train/labels/
            src_img_dir = dataset_path / split / 'images'
            src_lbl_dir = dataset_path / split / 'labels'

            if not src_img_dir.exists():
                continue

            dst_img_dir = merged_dir / 'images' / split
            dst_lbl_dir = merged_dir / 'labels' / split

            # Copy all images
            for img_file in src_img_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Rename to avoid conflicts
                    new_name = f"img_{file_counter:06d}{img_file.suffix}"
                    shutil.copy(img_file, dst_img_dir / new_name)

                    # Copy corresponding label
                    lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        shutil.copy(lbl_file, dst_lbl_dir / f"img_{file_counter:06d}.txt")

                    file_counter += 1

    # Create merged data.yaml
    merged_yaml = {
        'path': str(merged_dir.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': len(all_classes),
        'names': sorted(list(all_classes))
    }

    with open(merged_dir / 'data.yaml', 'w') as f:
        yaml.dump(merged_yaml, f, default_flow_style=False)

    # Print summary
    print(f"\n✅ Merged dataset created: {merged_dir}")
    print(f"   Classes: {len(all_classes)}")
    print(f"   Total images: {file_counter}")

    for split in ['train', 'valid', 'test']:
        count = len(list((merged_dir / 'images' / split).glob('*')))
        print(f"   {split}: {count} images")

    return merged_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download billiard ball datasets from Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge datasets into a single training set'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'datasets',
        help='Output directory for datasets (default: datasets/)'
    )

    args = parser.parse_args()

    print("="*70)
    print("  🎱 Roboflow Billiard Ball Dataset Downloader")
    print("="*70)

    # Get API key
    api_key = get_api_key()
    rf = Roboflow(api_key=api_key)

    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Define datasets to download
    # NOTE: Check dataset version on Roboflow website if download fails
    datasets = [
        {
            'workspace': 'billiardstrainer',
            'project': 'billiards-ta6xb',
            'version': 1,
            'name': 'billiardstrainer'
        }
    ]

    # Download all datasets
    downloaded_paths = []
    for ds in datasets:
        try:
            path = download_dataset(
                rf,
                ds['workspace'],
                ds['project'],
                ds['version'],
                args.output_dir,
                ds['name']
            )
            downloaded_paths.append(path)
        except Exception as e:
            print(f"⚠️  Skipping {ds['name']} due to error")
            continue

    if not downloaded_paths:
        print("\n❌ No datasets were downloaded successfully")
        return 1

    # Merge if requested
    if args.merge:
        merged_path = merge_datasets(downloaded_paths, args.output_dir)
        print(f"\n✅ Ready for training!")
        print(f"\nTo train with the merged dataset:")
        print(f"  python tools/train_yolo.py")
        print(f"\nOr specify the merged dataset path:")
        print(f"  # Update dataset path in train_yolo.py to: {merged_path / 'data.yaml'}")
    else:
        print(f"\n✅ Datasets downloaded to: {args.output_dir}")
        print(f"\nTo merge datasets:")
        print(f"  python tools/download_roboflow_datasets.py --merge")
        print(f"\nTo train with individual datasets:")
        for path in downloaded_paths:
            print(f"  python tools/train_yolo.py")
            print(f"  # Use dataset: {path / 'data.yaml'}")

    print("\n📚 Next steps:")
    print("  1. Review downloaded datasets")
    print("  2. Optionally merge datasets")
    print("  3. Run train_yolo.py to start training")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
