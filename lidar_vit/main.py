"""
Main entry point for the LIDAR Vision Transformer project.
This script demonstrates the complete workflow from raw LIDAR data to model training.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import our modules
from data.preprocessing import LIDARToViTTokenizer, preprocess_lidar_point_cloud


def main():
    """Main function demonstrating the LIDAR preprocessing pipeline."""
    print("LIDAR Vision Transformer - Complete Workflow")
    print("=" * 45)
    
    # Create sample LIDAR data (in practice, you would load real data)
    print("1. Generating sample LIDAR data...")
    # Simulate a point cloud with 50,000 points
    import numpy as np
    num_points = 50000
    points = np.random.rand(num_points, 4).astype(np.float32)
    points[:, 0] = points[:, 0] * 20 - 10  # x: -10 to 10 meters
    points[:, 1] = points[:, 1] * 50       # y: 0 to 50 meters (forward)
    points[:, 2] = points[:, 2] * 8 - 3    # z: -3 to 5 meters (ground at z=0)
    points[:, 3] = points[:, 3]            # intensity: 0 to 1
    
    print(f"   Generated {points.shape[0]} points with {points.shape[1]} features each")
    
    # Initialize the tokenizer
    print("\n2. Initializing LIDAR to ViT tokenizer...")
    tokenizer = LIDARToViTTokenizer()
    print(f"   Grid size: {tokenizer.grid_size}")
    print(f"   Voxel size: {tokenizer.voxel_size}")
    print(f"   Max points per voxel: {tokenizer.max_points_per_voxel}")
    
    # Process the point cloud
    print("\n3. Processing point cloud...")
    tokens, coords = tokenizer.forward(points)
    print(f"   Converted to {tokens.shape[0]} tokens with {tokens.shape[1]} features each")
    print(f"   Voxel coordinates shape: {coords.shape}")
    
    # Show some statistics
    import torch
    print("\n4. Token statistics:")
    print(f"   Mean token values: {torch.mean(tokens, dim=0)}")
    print(f"   Std token values: {torch.std(tokens, dim=0)}")
    
    # Demonstrate the convenience function
    print("\n5. Using convenience function...")
    tokens2, coords2 = preprocess_lidar_point_cloud(
        points,
        grid_size=(50, 50, 10),  # Different grid size for comparison
        voxel_size=(0.2, 0.2, 0.2)
    )
    print(f"   Converted to {tokens2.shape[0]} tokens with {tokens2.shape[1]} features each")
    
    print("\n6. Complete workflow demonstrated!")
    print("   The tokens are now ready to be fed into a Vision Transformer model.")
    
    # Show next steps
    print("\nAvailable scripts:")
    print("1. Training: python train_vit.py")
    print("2. Inference: python inference.py --model_path checkpoints/model_final.pth --input_data path/to/data.bin")
    print("3. Preprocessing examples: python -m data.example_preprocessing")
    print("4. Benchmarking: python -m data.benchmark_preprocessing")
    
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()