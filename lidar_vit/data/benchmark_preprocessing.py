"""
Benchmark script for LIDAR preprocessing functions.
"""

import time
import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import preprocessing module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import LIDARToViTTokenizer, preprocess_lidar_point_cloud


def benchmark_preprocessing():
    """Benchmark the LIDAR preprocessing pipeline."""
    print("Benchmarking LIDAR preprocessing functions...")
    
    # Create a realistic point cloud size (typical for LIDAR sensors)
    # A 64-channel LIDAR might produce 100k-200k points per scan
    point_counts = [10000, 50000, 100000, 200000]
    
    # Test different configurations
    configs = [
        {"grid_size": (100, 100, 20), "voxel_size": (0.1, 0.1, 0.1)},
        {"grid_size": (50, 50, 10), "voxel_size": (0.2, 0.2, 0.2)},
        {"grid_size": (200, 200, 20), "voxel_size": (0.05, 0.05, 0.05)},
    ]
    
    for num_points in point_counts:
        print(f"\n--- Testing with {num_points:,} points ---")
        
        # Generate test data
        points = np.random.rand(num_points, 4).astype(np.float32)
        points[:, 0] = points[:, 0] * 20 - 10  # x: -10 to 10
        points[:, 1] = points[:, 1] * 50       # y: 0 to 50
        points[:, 2] = points[:, 2] * 8 - 3    # z: -3 to 5
        points[:, 3] = points[:, 3]            # intensity: 0 to 1
        
        for config in configs:
            grid_size = config["grid_size"]
            voxel_size = config["voxel_size"]
            
            # Initialize tokenizer
            tokenizer = LIDARToViTTokenizer(
                grid_size=grid_size,
                voxel_size=voxel_size,
                max_points_per_voxel=32,
                normalize_coords=True,
                use_intensity=True
            )
            
            # Warm up
            for _ in range(3):
                _ = tokenizer.forward(points)
            
            # Benchmark
            times = []
            num_runs = 10
            
            for _ in range(num_runs):
                start_time = time.perf_counter()
                tokens, coords = tokenizer.forward(points)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"  Grid: {grid_size}, Voxel: {voxel_size}")
            print(f"    Avg time: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
            print(f"    Tokens generated: {tokens.shape[0]}")
            print(f"    Throughput: {num_points/(avg_time*1000):.0f}K points/sec")
            
            # Test convenience function as well
            start_time = time.perf_counter()
            tokens2, coords2 = preprocess_lidar_point_cloud(
                points,
                grid_size=grid_size,
                voxel_size=voxel_size,
                max_points_per_voxel=32
            )
            end_time = time.perf_counter()
            
            print(f"    Convenience function time: {(end_time-start_time)*1000:.2f} ms")


def benchmark_memory_usage():
    """Benchmark memory usage of preprocessing functions."""
    print("\n--- Memory Usage Benchmark ---")
    
    # Create a large point cloud
    num_points = 100000
    points = np.random.rand(num_points, 4).astype(np.float32)
    points[:, 0] = points[:, 0] * 20 - 10  # x: -10 to 10
    points[:, 1] = points[:, 1] * 50       # y: 0 to 50
    points[:, 2] = points[:, 2] * 8 - 3    # z: -3 to 5
    
    # Measure memory before
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
    else:
        mem_before = 0
        print("  CUDA not available, skipping GPU memory measurement")
    
    # Process point cloud
    tokens, coords = preprocess_lidar_point_cloud(
        points,
        grid_size=(100, 100, 20),
        voxel_size=(0.1, 0.1, 0.1),
        max_points_per_voxel=32
    )
    
    # Measure memory after
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        mem_used = (mem_after - mem_before) / (1024 * 1024)  # Convert to MB
        print(f"  GPU memory used: {mem_used:.2f} MB")
    
    print(f"  Input points: {points.shape}")
    print(f"  Output tokens: {tokens.shape}")
    print(f"  Compression ratio: {points.shape[0]/tokens.shape[0]:.2f}x")


if __name__ == "__main__":
    benchmark_preprocessing()
    benchmark_memory_usage()