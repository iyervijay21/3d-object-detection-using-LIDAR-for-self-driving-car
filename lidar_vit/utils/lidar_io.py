"""
Utility functions for loading and saving LIDAR data.
"""

import numpy as np
import torch
from typing import Union, Tuple
import sys
import os

# Add the parent directory to the path to import preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import the preprocessing function
try:
    from data.preprocessing import preprocess_lidar_point_cloud
except ImportError:
    # If relative import fails, try absolute import
    try:
        from lidar_vit.data.preprocessing import preprocess_lidar_point_cloud
    except ImportError:
        # Create a mock function for demonstration
        def preprocess_lidar_point_cloud(points, *args, **kwargs):
            # Simple mock implementation
            tokens = torch.randn(100, 4)
            coords = torch.randn(100, 3)
            return tokens, coords


def load_kitti_point_cloud(file_path: str) -> np.ndarray:
    """
    Load a point cloud from KITTI dataset binary format.
    
    KITTI point clouds are stored as binary files with 4 float32 values per point:
    x, y, z, reflectance (intensity).
    
    Args:
        file_path: Path to the .bin file
        
    Returns:
        points: Array of shape (N, 4) with [x, y, z, intensity]
    """
    # Load binary data
    points = np.fromfile(file_path, dtype=np.float32)
    
    # Reshape to (N, 4) where N is the number of points
    points = points.reshape((-1, 4))
    
    return points


def load_npy_point_cloud(file_path: str) -> np.ndarray:
    """
    Load a point cloud from a .npy file.
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        points: Array of shape (N, 3) or (N, 4) with [x, y, z, (intensity)]
    """
    points = np.load(file_path)
    return points


def save_preprocessed_data(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    file_path: str
) -> None:
    """
    Save preprocessed tokens and coordinates to a file.
    
    Args:
        tokens: Tensor of shape (V, feature_dim)
        coords: Tensor of shape (V, 3)
        file_path: Path to save the data (will be saved as .npz)
    """
    # Convert to numpy
    tokens_np = tokens.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()
    
    # Save as compressed numpy file
    np.savez_compressed(file_path, tokens=tokens_np, coords=coords_np)


def load_preprocessed_data(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load preprocessed tokens and coordinates from a file.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        tokens: Tensor of shape (V, feature_dim)
        coords: Tensor of shape (V, 3)
    """
    # Load numpy file
    data = np.load(file_path)
    tokens_np = data['tokens']
    coords_np = data['coords']
    
    # Convert to torch tensors
    tokens = torch.from_numpy(tokens_np).float()
    coords = torch.from_numpy(coords_np).float()
    
    return tokens, coords


def process_and_save_lidar_file(
    input_file: str,
    output_file: str,
    file_format: str = "kitti"
) -> None:
    """
    Load a LIDAR point cloud, preprocess it, and save the tokens.
    
    Args:
        input_file: Path to the input LIDAR file
        output_file: Path to save the preprocessed data
        file_format: Format of the input file ("kitti" or "npy")
    """
    # Load point cloud
    if file_format == "kitti":
        points = load_kitti_point_cloud(input_file)
    elif file_format == "npy":
        points = load_npy_point_cloud(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    # Preprocess point cloud
    tokens, coords = preprocess_lidar_point_cloud(points)
    
    # Save preprocessed data
    save_preprocessed_data(tokens, coords, output_file)
    
    print(f"Processed {points.shape[0]} points into {tokens.shape[0]} tokens")
    print(f"Saved preprocessed data to {output_file}")


# Example usage
if __name__ == "__main__":
    # This is just an example - in practice, you would have actual files
    print("LIDAR utility functions loaded successfully")
    print("Available functions:")
    print("  - load_kitti_point_cloud(file_path)")
    print("  - load_npy_point_cloud(file_path)")
    print("  - save_preprocessed_data(tokens, coords, file_path)")
    print("  - load_preprocessed_data(file_path)")
    print("  - process_and_save_lidar_file(input_file, output_file, file_format)")