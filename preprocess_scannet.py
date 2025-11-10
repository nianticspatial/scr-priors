import os
import argparse
import numpy as np
import open3d as o3d
import tqdm


def main():
    parser = argparse.ArgumentParser(description='Preprocess ScanNet point cloud data')
    parser.add_argument('--basedir', type=str, required=True,
                       help='Base directory containing ScanNet scans')
    parser.add_argument('--savedir', type=str, required=True,
                       help='Output directory to save processed point clouds')
    parser.add_argument('--n_downsample', type=int, default=500000,
                       help='Number of points to downsample to (default: 500000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible downsampling (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Validate input directory
    if not os.path.exists(args.basedir):
        raise ValueError(f"Base directory does not exist: {args.basedir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.savedir, exist_ok=True)
    
    # Get scene list
    scene_list = os.listdir(args.basedir)
    scene_list = [scene for scene in scene_list if 'scene' in scene]
    scene_list = sorted(scene_list)
    
    print(f"Found {len(scene_list)} scenes to process")
    print(f"Input directory: {args.basedir}")
    print(f"Output directory: {args.savedir}")
    print(f"Downsample size: {args.n_downsample}")
    print()
    
    for scene in tqdm.tqdm(scene_list, desc="Processing scenes"):
        pcd_path = os.path.join(args.basedir, scene, scene + '_vh_clean.ply')
        
        # Skip if PLY file doesn't exist
        if not os.path.exists(pcd_path):
            print(f"Warning: {pcd_path} not found, skipping {scene}")
            continue
            
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        # Downsample if needed
        if points.shape[0] > args.n_downsample:
            idx = np.random.choice(points.shape[0], args.n_downsample, replace=False)
            points = points[idx]
        
        # Create scene output directory
        scene_path = os.path.join(args.savedir, scene)
        os.makedirs(scene_path, exist_ok=True)
        
        # Save to NPZ
        npz_path = os.path.join(scene_path, 'point_cloud.npz')
        np.savez(npz_path, points=points)
    
    print(f"Processing complete! Results saved to: {args.savedir}")


if __name__ == "__main__":
    main()

