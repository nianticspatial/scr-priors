"""Setup Indoor-6 dataset.

See: https://github.com/microsoft/SceneLandmarkLocalization for more information.
"""
import argparse
import pathlib
import gdown
import shutil
import os
import multiprocessing
import numpy as np

# options
KEEP_RAW = True
SKIP_EXISTING = True

def parse_arguments():
    parser = argparse.ArgumentParser(description='Setup Indoor-6 dataset from Google Drive sources')
    parser.add_argument('--raw_dir', type=str, default='./indoor6_raw/',
                       help='Directory to download raw zip files (default: ./indoor6_raw/)')
    parser.add_argument('--target_dir', type=str, default='./indoor6/',
                       help='Target directory for processed dataset (default: ./indoor6/)')
    return parser.parse_args()

def readlines(p):
    with open(p, "r") as f:
        return f.readlines()

def setup_scene(data):
    source_url, raw_dir, target_dir = data
    print(f"Processing {source_url}...")
    zip_path = gdown.download(source_url, output=raw_dir, resume=SKIP_EXISTING)
    scene_name = pathlib.Path(zip_path).stem

    if os.path.exists(f"{target_dir}/{scene_name}"):
        print(f"Scene {scene_name} already exists. Skipping.")
        return

    os.system(f"unzip -qq -o {zip_path} -d {raw_dir}")

    # create target directories
    os.makedirs(f"{target_dir}/{scene_name}/train/rgb", exist_ok=True)
    os.makedirs(f"{target_dir}/{scene_name}/train/poses", exist_ok=True)
    os.makedirs(f"{target_dir}/{scene_name}/train/calibration", exist_ok=True)

    os.makedirs(f"{target_dir}/{scene_name}/val/rgb", exist_ok=True)
    os.makedirs(f"{target_dir}/{scene_name}/val/poses", exist_ok=True)
    os.makedirs(f"{target_dir}/{scene_name}/val/calibration", exist_ok=True)

    os.makedirs(f"{target_dir}/{scene_name}/test/rgb", exist_ok=True)
    os.makedirs(f"{target_dir}/{scene_name}/test/poses", exist_ok=True)
    os.makedirs(f"{target_dir}/{scene_name}/test/calibration", exist_ok=True)

    # move or copy files
    copy_or_move = shutil.copyfile if KEEP_RAW else os.rename

    src_dir = pathlib.Path(raw_dir) / scene_name
    src_data_dir = src_dir / "images"

    for split in ["train", "val", "test"]:
        rgb_files = readlines(src_dir / f"{scene_name}_{split}.txt")
        for rgb_file in rgb_files:
            rgb_file = rgb_file.strip()
            stem = rgb_file.split(".")[0]
            copy_or_move(src_data_dir / rgb_file, f"{target_dir}/{scene_name}/{split}/rgb/{stem}.jpg")
            copy_or_move(src_data_dir / f"{stem}.pose.txt", f"{target_dir}/{scene_name}/{split}/poses/{stem}.txt")
            raw_w2c_34 = np.loadtxt(src_data_dir / f"{stem}.pose.txt")
            c2w_44 = np.linalg.inv(np.vstack([raw_w2c_34, [[0, 0, 0, 1]]]))
            np.savetxt(f"{target_dir}/{scene_name}/{split}/poses/{stem}.txt", c2w_44)
            copy_or_move(src_data_dir / f"{stem}.intrinsics.txt", f"{target_dir}/{scene_name}/{split}/calibration/{stem}.txt")


if __name__ == "__main__":
    args = parse_arguments()

    print("\n############################################################################")
    print("# Please make sure to check this dataset's license before using it!        #")
    print("# https://github.com/microsoft/SceneLandmarkLocalization                   #")
    print("############################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.target_dir, exist_ok=True)

    source_urls = [
        "https://drive.google.com/uc?id=1AJhPh9nnZO0HJyxuXXZdtKtA7kFRi3LQ", # scene1
        "https://drive.google.com/uc?id=1DgTQ7fflZJ7DdbHDRZF-6gXdB_vJF7fY", # scene2a
        "https://drive.google.com/uc?id=12aER7rQkvGS_DPeugTHo_Ma_Fi7JuflS", # scene3
        "https://drive.google.com/uc?id=1gibneq5ixZ0lmeNAYTmY4Mh8a244T2nl", # scene4a
        "https://drive.google.com/uc?id=18wHn_69-eV22N4I8R0rWQkcSQ3EtCYMX", # scene5
        "https://drive.google.com/uc?id=1mZYnoKo37KXRjREK5CKs5IzDox2G3Prt", # scene6
    ]

    # Prepare data for multiprocessing (each worker needs the source_url, raw_dir, and target_dir)
    worker_data = [(url, args.raw_dir, args.target_dir) for url in source_urls]

    with multiprocessing.Pool(8) as pool:
        pool.map(setup_scene, worker_data)
