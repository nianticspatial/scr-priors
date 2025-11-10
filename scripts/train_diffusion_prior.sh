#!/bin/bash

# Define paths

# Point to the scans folder within the scannet download
SCANNET_DIR="./datasets/scannet/scans"

# Folder to contain the preprocessed scannet files for model training (subsampled point clouds)
SCANNET_PCD_DIR="./datasets/scannet_pcd"

# Output directory containing model checkpoints and logs
OUTPUT_DIR="out_diffusion_model"

# Preprocess only if output directory doesn't exist
if [ ! -d "$SCANNET_PCD_DIR" ]; then
    echo "Preprocessing ScanNet data..."
    python preprocess_scannet.py \
            --basedir "$SCANNET_DIR" \
            --savedir "$SCANNET_PCD_DIR"
else
    echo "Preprocessed data already exists at $SCANNET_PCD_DIR, skipping preprocessing."
fi

python train_diffusion_prior.py \
        --max_iters 100000 \
        --tag pvcnn_allscene_T_20_lr \
        --ckpt_freq 10000 \
        --batch_size 16 \
        --dataset_path "$SCANNET_PCD_DIR" \
        --log_root "$OUTPUT_DIR" \
        --lr 2e-4 \
        --end_lr 0.0 \
        --sched_start_epoch 1000 \
        --sched_end_epoch 100000 \
        --n_points 5120 \
        --rotate True