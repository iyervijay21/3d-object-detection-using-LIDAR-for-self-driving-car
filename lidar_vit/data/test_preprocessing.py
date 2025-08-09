"""
Unit tests for LIDAR preprocessing functions.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import preprocessing module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import LIDARToViTTokenizer, preprocess_lidar_point_cloud


class TestLIDARToViTTokenizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = LIDARToViTTokenizer()
        # Create sample point cloud data
        self.sample_points = np.array([
            [1.0, 2.0, 0.5, 0.8],  # x, y, z, intensity
            [-1.0, 3.0, 1.0, 0.6],
            [0.5, 1.5, 0.2, 0.9],
            [2.0, 4.0, 1.5, 0.7],
            [-0.5, 2.5, 0.8, 0.5],
        ], dtype=np.float32)
    
    def test_voxelization(self):
        """Test the voxelization process."""
        points_tensor = torch.from_numpy(self.sample_points)
        voxel_features, voxel_coords, counts = self.tokenizer.voxelize_point_cloud(points_tensor)
        
        # Check output shapes
        self.assertEqual(voxel_features.dim(), 3)
        self.assertEqual(voxel_coords.dim(), 2)
        self.assertEqual(counts.dim(), 1)
        
        # Check that we have consistent number of voxels
        self.assertEqual(voxel_features.shape[0], voxel_coords.shape[0])
        self.assertEqual(voxel_features.shape[0], counts.shape[0])
    
    def test_feature_normalization(self):
        """Test feature normalization."""
        points_tensor = torch.from_numpy(self.sample_points)
        voxel_features, voxel_coords, _ = self.tokenizer.voxelize_point_cloud(points_tensor)
        
        # Normalize features
        normalized_features = self.tokenizer.normalize_features(voxel_features, voxel_coords)
        
        # Check that coordinates are in [0, 1] range when normalization is enabled
        if self.tokenizer.normalize_coords:
            self.assertTrue(torch.all(normalized_features[:, :, 0] >= 0))
            self.assertTrue(torch.all(normalized_features[:, :, 0] <= 1))
            self.assertTrue(torch.all(normalized_features[:, :, 1] >= 0))
            self.assertTrue(torch.all(normalized_features[:, :, 1] <= 1))
            self.assertTrue(torch.all(normalized_features[:, :, 2] >= 0))
            self.assertTrue(torch.all(normalized_features[:, :, 2] <= 1))
    
    def test_feature_aggregation(self):
        """Test feature aggregation within voxels."""
        points_tensor = torch.from_numpy(self.sample_points)
        voxel_features, voxel_coords, _ = self.tokenizer.voxelize_point_cloud(points_tensor)
        normalized_features = self.tokenizer.normalize_features(voxel_features, voxel_coords)
        
        # Aggregate features
        aggregated_features = self.tokenizer.aggregate_voxel_features(normalized_features)
        
        # Check output shape
        self.assertEqual(aggregated_features.dim(), 2)
        self.assertEqual(aggregated_features.shape[0], voxel_features.shape[0])
        self.assertEqual(aggregated_features.shape[1], self.tokenizer.feature_dim)
    
    def test_full_pipeline(self):
        """Test the full preprocessing pipeline."""
        tokens, coords = self.tokenizer.forward(self.sample_points)
        
        # Check output shapes
        self.assertEqual(tokens.dim(), 2)
        self.assertEqual(coords.dim(), 2)
        self.assertEqual(tokens.shape[1], self.tokenizer.feature_dim)
        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(tokens.shape[0], coords.shape[0])
    
    def test_convenience_function(self):
        """Test the convenience preprocessing function."""
        tokens, coords = preprocess_lidar_point_cloud(self.sample_points)
        
        # Check output shapes
        self.assertEqual(tokens.dim(), 2)
        self.assertEqual(coords.dim(), 2)
        self.assertEqual(tokens.shape[1], 4)  # x, y, z, intensity
        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(tokens.shape[0], coords.shape[0])


class TestEdgeCases(unittest.TestCase):
    def test_empty_point_cloud(self):
        """Test handling of empty point clouds."""
        tokenizer = LIDARToViTTokenizer()
        empty_points = np.empty((0, 4), dtype=np.float32)
        
        tokens, coords = tokenizer.forward(empty_points)
        
        # Should return empty tensors
        self.assertEqual(tokens.shape[0], 0)
        self.assertEqual(coords.shape[0], 0)
    
    def test_out_of_bounds_points(self):
        """Test handling of points outside the grid."""
        tokenizer = LIDARToViTTokenizer()
        # Create points outside the default grid range
        out_of_bounds_points = np.array([
            [100.0, 100.0, 100.0, 0.5],  # Way outside
            [-100.0, -100.0, -100.0, 0.5],  # Way outside
        ], dtype=np.float32)
        
        tokens, coords = tokenizer.forward(out_of_bounds_points)
        
        # Should return empty tensors since all points are out of bounds
        self.assertEqual(tokens.shape[0], 0)
        self.assertEqual(coords.shape[0], 0)


if __name__ == "__main__":
    unittest.main()