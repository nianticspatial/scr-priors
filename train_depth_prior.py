import argparse
import glob
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import laplace
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fit Laplace distribution to depth values from depth files.")
    parser.add_argument("glob_pattern", type=str, 
                       help="Glob pattern to match depth files (PGM or 16-bit PNG), e.g. '/path/to/scenes/*/sensor_data/*.pgm'")
    parser.add_argument("--plot", action="store_true",
                       help="Generate histogram plot with fitted Laplace distribution")
    parser.add_argument("--output_png", type=str, default="depth_laplace_fit.png",
                       help="Output PNG filename for plot (only used with --plot)")
    parser.add_argument("--num_files", type=int, default=100,
                       help="Maximum number of depth files to sample and process (default: 100)")
    parser.add_argument("--num_bins", type=int, default=256,
                       help="Number of histogram bins for plotting (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible file sampling (default: 42)")
    return parser.parse_args()


def read_depth_file(filepath):
    """Reads a depth file (PGM or 16-bit PNG) and returns a NumPy array."""
    try:
        img = Image.open(filepath)
        depth_array = np.array(img)
        
        # Handle 16-bit PNGs - they should already be in the correct format
        # PGMs are typically already in the right format too
        if depth_array.dtype == np.uint16:
            # Keep as uint16 for now, will convert to meters later
            return depth_array
        elif depth_array.dtype == np.uint8:
            # Convert 8-bit to 16-bit if needed
            return depth_array.astype(np.uint16) * 256
        else:
            return depth_array.astype(np.uint16)
            
    except Exception as e:
        logging.error(f"Error reading depth file {filepath}: {e}")
        return None


def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducible file sampling
    random.seed(args.seed)

    try:
        # Find all depth files matching the glob pattern
        all_depth_files = glob.glob(args.glob_pattern)
        if not all_depth_files:
            raise ValueError(f"No files found matching pattern: {args.glob_pattern}")
        
        logger.info(f"Found {len(all_depth_files)} depth files matching pattern")
        
        # Sample depth files
        num_files_to_sample = min(args.num_files, len(all_depth_files))
        if num_files_to_sample < len(all_depth_files):
            depth_files = random.sample(all_depth_files, num_files_to_sample)
            logger.info(f"Randomly sampled {num_files_to_sample} files from {len(all_depth_files)} available files")
        else:
            depth_files = all_depth_files
            logger.info(f"Processing all {num_files_to_sample} available files")
        
        all_depth_values = []
        successful_files = 0
        
        # Load depth values from sampled files with progress bar
        logger.info("Loading depth data from files...")
        for depth_file in tqdm(depth_files, desc="Processing files", unit="file"):
            depth_map = read_depth_file(depth_file)
            if depth_map is not None:
                all_depth_values.extend(depth_map.flatten())
                successful_files += 1
            else:
                logger.warning(f"Failed to load: {depth_file}")
        
        if not all_depth_values:
            raise ValueError("No valid depth data found in any files.")
        
        logger.info(f"Successfully loaded depth data from {successful_files}/{len(depth_files)} files")
        
        # Convert to meters and exclude zero values
        depth_values_meters = np.array(all_depth_values) / 1000.0  # Convert mm to m
        depth_values_meters = depth_values_meters[depth_values_meters > 0]  # Remove zero values
        
        logger.info(f"Collected {len(depth_values_meters):,} valid depth values (non-zero)")
        
        if not depth_values_meters.size:
            raise ValueError("No valid depth data found after filtering zero values.")
        
        # Fit Laplace distribution (using all depth values from sampled files)
        logger.info("Fitting Laplace distribution...")
        loc_laplace, scale_laplace = laplace.fit(depth_values_meters)
        
        # Log and print results
        logger.info("Laplace distribution fitting completed")
        logger.info(f"Location (loc): {loc_laplace:.6f} meters")
        logger.info(f"Scale (scale): {scale_laplace:.6f} meters")
        logger.info(f"Data range: {depth_values_meters.min():.3f} - {depth_values_meters.max():.3f} meters")
        logger.info(f"Total depth values used for fitting: {len(depth_values_meters):,}")
        logger.info(f"Files processed: {successful_files}/{len(depth_files)}")
        
        # Also print key results to console for easy access
        print(f"\nLaplace Distribution Parameters:")
        print(f"Location (loc): {loc_laplace:.3f} meters")
        print(f"Scale (scale): {scale_laplace:.3f} meters")
        
        # Optional plotting
        if args.plot:
            logger.info("Generating histogram plot with fitted Laplace distribution...")
            
            # Create output directory if needed
            import os
            output_dir = os.path.dirname(args.output_png)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.hist(depth_values_meters, bins=args.num_bins, density=True, alpha=0.7, 
                    label="Depth Data", color='skyblue', edgecolor='black')
            
            # Plot fitted Laplace distribution
            x_range = np.linspace(depth_values_meters.min(), depth_values_meters.max(), 1000)
            laplace_y = laplace.pdf(x_range, loc_laplace, scale_laplace)
            plt.plot(x_range, laplace_y, 'r-', linewidth=2, 
                    label=f"Laplace Fit (loc={loc_laplace:.3f}, scale={scale_laplace:.3f})")
            
            plt.title(f"Depth Value Distribution with Fitted Laplace Distribution\n({successful_files} files, {len(depth_values_meters):,} depth values)")
            plt.xlabel("Depth Value (meters)")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(args.output_png, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to: {args.output_png}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()