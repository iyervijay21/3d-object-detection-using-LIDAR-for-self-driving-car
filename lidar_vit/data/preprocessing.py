"""
Data preprocessing functions for converting LIDAR point clouds to Vision Transformer input tokens.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import torch.nn.functional as F
import sys
import os

# Add the parent directory to the path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.preprocessing_config import DEFAULT_PREPROCESSING_CONFIG


class LIDARToViTTokenizer:
    """
    Converts raw LIDAR point clouds into Vision Transformer compatible input tokens.
    
    This class handles the conversion of 3D point clouds into a format suitable
    for Vision Transformer models, including voxelization, normalization, and tokenization.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = tuple(DEFAULT_PREPROCESSING_CONFIG["grid_size"]),
        voxel_size: Tuple[float, float, float] = tuple(DEFAULT_PREPROCESSING_CONFIG["voxel_size"]),
        max_points_per_voxel: int = DEFAULT_PREPROCESSING_CONFIG["max_points_per_voxel"],
        normalize_coords: bool = DEFAULT_PREPROCESSING_CONFIG["normalize_coords"],
        use_intensity: bool = DEFAULT_PREPROCESSING_CONFIG["use_intensity"],
    ):
        """
        Initialize the LIDAR to ViT tokenizer.
        
        Args:
            grid_size: Number of voxels in (x, y, z) dimensions
            voxel_size: Size of each voxel in meters for (x, y, z) dimensions
            max_points_per_voxel: Maximum number of points per voxel
            normalize_coords: Whether to normalize coordinates to [0, 1]
            use_intensity: Whether to include intensity values in the features
        """
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.normalize_coords = normalize_coords
        self.use_intensity = use_intensity
        self.feature_dim = 4 if use_intensity else 3  # x, y, z, [intensity]
        
        # Calculate the physical boundaries based on grid and voxel size
        self.x_range = (-grid_size[0] // 2 * voxel_size[0], grid_size[0] // 2 * voxel_size[0])
        self.y_range = (0, grid_size[1] * voxel_size[1])  # Forward direction
        self.z_range = (-3, grid_size[2] * voxel_size[2] - 3)  # Ground is at z=0
        
    def voxel_coords_to_indices(
        self, 
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert point coordinates to voxel indices.
        
        Args:
            points: Tensor of shape (N, 3) or (N, 4) containing [x, y, z, (intensity)]
            
        Returns:
            voxel_indices: Tensor of shape (N, 3) with voxel indices for each point
            valid_mask: Boolean tensor of shape (N,) indicating valid points
        """
        # Extract coordinates
        coords = points[:, :3]  # Take x, y, z coordinates
        
        # Calculate voxel indices
        # Shift coordinates to positive range
        x_indices = torch.floor((coords[:, 0] - self.x_range[0]) / self.voxel_size[0]).long()
        y_indices = torch.floor((coords[:, 1] - self.y_range[0]) / self.voxel_size[1]).long()
        z_indices = torch.floor((coords[:, 2] - self.z_range[0]) / self.voxel_size[2]).long()
        
        # Create voxel indices tensor
        voxel_indices = torch.stack([x_indices, y_indices, z_indices], dim=1)
        
        # Create validity mask
        valid_mask = (
            (x_indices >= 0) & (x_indices < self.grid_size[0]) &
            (y_indices >= 0) & (y_indices < self.grid_size[1]) &
            (z_indices >= 0) & (z_indices < self.grid_size[2])
        )
        
        return voxel_indices, valid_mask
    
    def voxelize_point_cloud(
        self, 
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Voxelize a point cloud and aggregate points within each voxel.
        
        Args:
            points: Tensor of shape (N, 3) or (N, 4) containing [x, y, z, (intensity)]
            
        Returns:
            voxel_features: Tensor of shape (V, max_points_per_voxel, feature_dim)
            voxel_coords: Tensor of shape (V, 3) with voxel coordinates
            voxel_counts: Tensor of shape (V,) with number of points in each voxel
        """
        # Get voxel indices for all points
        voxel_indices, valid_mask = self.voxel_coords_to_indices(points)
        
        # Filter valid points
        valid_points = points[valid_mask]
        valid_voxel_indices = voxel_indices[valid_mask]
        
        if valid_points.shape[0] == 0:
            # Return empty tensors if no valid points
            return (
                torch.empty(0, self.max_points_per_voxel, self.feature_dim),
                torch.empty(0, 3),
                torch.empty(0)
            )
        
        # Convert 3D voxel indices to 1D indices for grouping
        one_d_indices = (
            valid_voxel_indices[:, 0] * self.grid_size[1] * self.grid_size[2] +
            valid_voxel_indices[:, 1] * self.grid_size[2] +
            valid_voxel_indices[:, 2]
        )
        
        # Sort points by voxel indices
        sorted_indices = torch.argsort(one_d_indices)
        sorted_points = valid_points[sorted_indices]
        sorted_voxel_indices = valid_voxel_indices[sorted_indices]
        sorted_one_d_indices = one_d_indices[sorted_indices]
        
        # Find unique voxels and their counts
        unique_voxels, inverse_indices, counts = torch.unique(
            sorted_one_d_indices, return_inverse=True, return_counts=True
        )
        
        # Pad or truncate points in each voxel
        max_points = self.max_points_per_voxel
        total_voxels = unique_voxels.shape[0]
        
        # Initialize output tensors
        voxel_features = torch.zeros(total_voxels, max_points, self.feature_dim)
        voxel_coords = torch.zeros(total_voxels, 3)
        
        # Process each unique voxel
        start_idx = 0
        for i, (voxel_idx, count) in enumerate(zip(unique_voxels, counts)):
            end_idx = start_idx + count
            
            # Get points in this voxel
            voxel_points = sorted_points[start_idx:end_idx]
            
            # Pad or truncate to max_points_per_voxel
            if count > max_points:
                # Randomly sample points if we have too many
                indices = torch.randperm(count)[:max_points]
                voxel_points = voxel_points[indices]
                count = max_points
            elif count < max_points:
                # Pad with zeros if we have too few
                padding = torch.zeros(max_points - count, voxel_points.shape[1])
                voxel_points = torch.cat([voxel_points, padding], dim=0)
            
            # Store features and coordinates
            if self.use_intensity:
                voxel_features[i] = voxel_points[:, :self.feature_dim]  # x, y, z, intensity
            else:
                voxel_features[i] = voxel_points[:, :self.feature_dim]  # x, y, z
            
            # Convert 1D index back to 3D coordinates
            x = (voxel_idx // (self.grid_size[1] * self.grid_size[2])).long()
            y = ((voxel_idx % (self.grid_size[1] * self.grid_size[2])) // self.grid_size[2]).long()
            z = (voxel_idx % self.grid_size[2]).long()
            voxel_coords[i] = torch.tensor([x, y, z])
            
            start_idx = end_idx
        
        return voxel_features, voxel_coords, counts
    
    def normalize_features(
        self, 
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize voxel features for better training stability.
        
        Args:
            voxel_features: Tensor of shape (V, max_points_per_voxel, feature_dim)
            voxel_coords: Tensor of shape (V, 3) with voxel coordinates
            
        Returns:
            normalized_features: Tensor of shape (V, max_points_per_voxel, feature_dim)
        """
        if voxel_features.shape[0] == 0:
            return voxel_features
            
        # Normalize coordinates to [0, 1] range
        if self.normalize_coords:
            # Normalize x coordinate
            x_min, x_max = self.x_range
            voxel_features[:, :, 0] = (voxel_features[:, :, 0] - x_min) / (x_max - x_min)
            
            # Normalize y coordinate
            y_min, y_max = self.y_range
            voxel_features[:, :, 1] = (voxel_features[:, :, 1] - y_min) / (y_max - y_min)
            
            # Normalize z coordinate
            z_min, z_max = self.z_range
            voxel_features[:, :, 2] = (voxel_features[:, :, 2] - z_min) / (z_max - z_min)
        
        # Normalize intensity if present
        if self.use_intensity and voxel_features.shape[2] > 3:
            # Clamp intensity values to reasonable range and normalize
            voxel_features[:, :, 3] = torch.clamp(voxel_features[:, :, 3], 0, 1)
        
        return voxel_features
    
    def aggregate_voxel_features(
        self, 
        voxel_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate features within each voxel using max pooling.
        
        Args:
            voxel_features: Tensor of shape (V, max_points_per_voxel, feature_dim)
            
        Returns:
            aggregated_features: Tensor of shape (V, feature_dim)
        """
        if voxel_features.shape[0] == 0:
            return torch.empty(0, self.feature_dim)
            
        # Use max pooling to aggregate features within each voxel
        aggregated_features = torch.max(voxel_features, dim=1)[0]
        return aggregated_features
    
    def forward(
        self, 
        points: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw LIDAR points to ViT-compatible tokens.
        
        Args:
            points: Array of shape (N, 3) or (N, 4) containing [x, y, z, (intensity)]
            
        Returns:
            tokens: Tensor of shape (V, feature_dim) with one token per voxel
            coords: Tensor of shape (V, 3) with voxel coordinates
        """
        # Convert numpy array to tensor if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        
        # Voxelize point cloud
        voxel_features, voxel_coords, _ = self.voxelize_point_cloud(points)
        
        # Normalize features
        voxel_features = self.normalize_features(voxel_features, voxel_coords)
        
        # Aggregate features within each voxel
        tokens = self.aggregate_voxel_features(voxel_features)
        
        return tokens, voxel_coords


def preprocess_lidar_point_cloud(
    points: Union[np.ndarray, torch.Tensor],
    grid_size: Tuple[int, int, int] = tuple(DEFAULT_PREPROCESSING_CONFIG["grid_size"]),
    voxel_size: Tuple[float, float, float] = tuple(DEFAULT_PREPROCESSING_CONFIG["voxel_size"]),
    max_points_per_voxel: int = DEFAULT_PREPROCESSING_CONFIG["max_points_per_voxel"],
    normalize_coords: bool = DEFAULT_PREPROCESSING_CONFIG["normalize_coords"],
    use_intensity: bool = DEFAULT_PREPROCESSING_CONFIG["use_intensity"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to preprocess LIDAR point cloud data for Vision Transformer.
    
    Args:
        points: Array of shape (N, 3) or (N, 4) containing [x, y, z, (intensity)]
        grid_size: Number of voxels in (x, y, z) dimensions
        voxel_size: Size of each voxel in meters for (x, y, z) dimensions
        max_points_per_voxel: Maximum number of points per voxel
        normalize_coords: Whether to normalize coordinates to [0, 1]
        use_intensity: Whether to include intensity values in the features
        
    Returns:
        tokens: Tensor of shape (V, feature_dim) with one token per voxel
        coords: Tensor of shape (V, 3) with voxel coordinates
    """
    # Initialize tokenizer
    tokenizer = LIDARToViTTokenizer(
        grid_size=grid_size,
        voxel_size=voxel_size,
        max_points_per_voxel=max_points_per_voxel,
        normalize_coords=normalize_coords,
        use_intensity=use_intensity,
    )
    
    # Process point cloud
    tokens, coords = tokenizer.forward(points)
    
    return tokens, coords