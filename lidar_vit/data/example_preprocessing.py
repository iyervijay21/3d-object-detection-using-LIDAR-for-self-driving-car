"""
Example script demonstrating how to use the LIDAR preprocessing functions
for Vision Transformer-based object detection.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import preprocessing module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import LIDARToViTTokenizer, preprocess_lidar_point_cloud

# Check if we have the required dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not found. Visualization will be skipped.")


def generate_sample_lidar_data(num_points: int = 1000) -> np.ndarray:
    """
    Generate sample LIDAR point cloud data for demonstration.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        points: Array of shape (N, 4) with [x, y, z, intensity]
    """
    # Generate random points in a reasonable range for autonomous driving
    # X: -10 to 10 meters (left/right)
    # Y: 0 to 50 meters (forward)
    # Z: -3 to 5 meters (down/up)
    x = np.random.uniform(-10, 10, num_points)
    y = np.random.uniform(0, 50, num_points)
    z = np.random.uniform(-3, 5, num_points)
    
    # Simulate intensity values
    intensity = np.random.uniform(0, 1, num_points)
    
    # Stack into point cloud array
    points = np.stack([x, y, z, intensity], axis=1)
    
    return points


def visualize_point_cloud(points: np.ndarray, title: str = "LIDAR Point Cloud"):
    """
    Visualize a 3D point cloud.
    
    Args:
        points: Array of shape (N, 3) or (N, 4) with [x, y, z, (intensity)]
        title: Title for the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualization - matplotlib not available")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Use intensity for color if available
    if points.shape[1] > 3:
        colors = points[:, 3]
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1)
        plt.colorbar(scatter, ax=ax, label='Intensity')
    else:
        ax.scatter(x, y, z, s=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    plt.show()


def demonstrate_preprocessing():
    """Demonstrate the LIDAR preprocessing pipeline."""
    print("Generating sample LIDAR data...")
    points = generate_sample_lidar_data(5000)
    print(f"Generated {points.shape[0]} points with {points.shape[1]} features each")
    
    # Visualize original point cloud
    print("Visualizing original point cloud...")
    visualize_point_cloud(points, "Original LIDAR Point Cloud")
    
    # Initialize tokenizer
    print("Initializing LIDAR to ViT tokenizer...")
    tokenizer = LIDARToViTTokenizer(
        grid_size=(100, 100, 20),  # 100x100x20 voxels
        voxel_size=(0.2, 0.2, 0.2),  # 0.2m resolution in each dimension
        max_points_per_voxel=32,
        normalize_coords=True,
        use_intensity=True
    )
    
    # Process point cloud
    print("Processing point cloud...")
    tokens, coords = tokenizer.forward(points)
    
    print(f"Converted to {tokens.shape[0]} tokens with {tokens.shape[1]} features each")
    print(f"Token features: {tokens.shape}")
    print(f"Voxel coordinates: {coords.shape}")
    
    # Show some statistics
    print("\nToken statistics:")
    print(f"  Mean token values: {torch.mean(tokens, dim=0)}")
    print(f"  Std token values: {torch.std(tokens, dim=0)}")
    
    # Demonstrate convenience function
    print("\nUsing convenience function...")
    tokens2, coords2 = preprocess_lidar_point_cloud(
        points,
        grid_size=(50, 50, 10),  # Smaller grid for comparison
        voxel_size=(0.4, 0.4, 0.4),  # Larger voxels
        max_points_per_voxel=16
    )
    
    print(f"Converted to {tokens2.shape[0]} tokens with {tokens2.shape[1]} features each")
    
    # Visualize voxel centers
    print("Visualizing voxel centers...")
    if MATPLOTLIB_AVAILABLE:
        # Convert voxel indices back to physical coordinates
        x_coords = coords2[:, 0] * 0.4 - 10  # Adjust for grid parameters
        y_coords = coords2[:, 1] * 0.4
        z_coords = coords2[:, 2] * 0.4 - 3
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, s=20, alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Voxel Centers')
        plt.show()
    else:
        print("Skipping visualization - matplotlib not available")


if __name__ == "__main__":
    demonstrate_preprocessing()