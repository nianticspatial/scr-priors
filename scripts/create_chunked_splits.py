#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import json
import logging
import argparse
import os
from pathlib import Path
import glob

_logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Creat benchmarking train/test split files for ScanNet. '
                    'Takes a chunk of N images every 2N images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_root', type=Path, help="Root folder of the dataset.")

    parser.add_argument('--test_step', type=int, default=60, help="Select N test images every 2N images")

    parser.add_argument('output_folder', type=Path, help="Where to store the split files.")

    args = parser.parse_args()

    # Create the output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Scenes as sub folders of the dataset root
    scene_folders = [f for f in args.dataset_root.glob('*/') if f.is_dir()]

    # Process each scene
    for scene_folder in scene_folders:

        _logger.info(f"Processing scene {scene_folder.name}.")

        # Get the image files
        scene_image_folder = scene_folder / "sensor_data"
        image_files = sorted(list(glob.glob(f"{scene_image_folder}/*.color.jpg")))

        # Split by taking n items 2xn items for test, and everything else for train
        n = args.test_step
        test_files = [item for i in range(0, len(image_files), 2 * n) if i + n <= len(image_files) for item in image_files[i:i + n]]
        train_files = [f for f in image_files if f not in test_files]

        _logger.info(f"Found {len(test_files)} test files and {len(train_files)} train files.")

        # Create the split info
        split_info = {
            'train_filenames': train_files,
            'test_filenames': test_files
        }

        # Store the split info in a JSON file
        _logger.info(f"Writing split info to {args.output_folder / f'{scene_folder.name}.json'}")

        with open(args.output_folder / f"{scene_folder.name}.json", 'w') as f:
            json.dump(split_info, f)
