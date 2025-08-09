"""
Inference script for LIDAR Vision Transformer obstacle detection model.
This script demonstrates how to load a trained model and run inference on LIDAR data.
"""

import torch
import numpy as np
import argparse
import os
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lidar_vit'))

# Import our modules
from data.preprocessing import LIDARToViTTokenizer, preprocess_lidar_point_cloud
from models.vit_model import LIDARVisionTransformer
from utils.lidar_io import load_kitti_point_cloud, load_npy_point_cloud


def load_trained_model(model_path: str, device: torch.device) -> LIDARVisionTransformer:
    """
    Load a trained LIDAR Vision Transformer model.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded LIDAR Vision Transformer model
    """
    # Create model instance
    model = LIDARVisionTransformer(
        num_classes=2,  # Background and obstacle
        feature_dim=4   # x, y, z, intensity
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def run_inference(
    model: LIDARVisionTransformer,
    points: np.ndarray,
    device: torch.device
) -> dict:
    """
    Run inference on LIDAR point cloud data.
    
    Args:
        model: Trained LIDAR Vision Transformer model
        points: LIDAR point cloud data of shape (N, 3) or (N, 4)
        device: Device to run inference on
        
    Returns:
        results: Dictionary containing inference results
    """
    # Preprocess point cloud
    tokens, coords = preprocess_lidar_point_cloud(points)
    
    # Add batch dimension
    tokens = tokens.unsqueeze(0)  # (1, N, 4)
    coords = coords.unsqueeze(0)  # (1, N, 3)
    
    # Move to device
    tokens = tokens.to(device)
    coords = coords.to(device)
    
    # Run inference
    with torch.no_grad():
        class_logits, bbox_preds = model(tokens, coords)
        
        # Apply softmax to get class probabilities
        class_probs = torch.softmax(class_logits, dim=-1)
        
        # Get predicted classes
        pred_classes = torch.argmax(class_probs, dim=-1)
    
    # Move results back to CPU
    class_probs = class_probs.cpu()
    pred_classes = pred_classes.cpu()
    bbox_preds = bbox_preds.cpu()
    
    # Prepare results
    results = {
        "tokens": tokens.cpu(),
        "coordinates": coords.cpu(),
        "class_probabilities": class_probs,
        "predicted_classes": pred_classes,
        "bounding_boxes": bbox_preds
    }
    
    return results


def visualize_results(points: np.ndarray, results: dict) -> None:
    """
    Visualize inference results (placeholder function).
    
    Args:
        points: Original LIDAR point cloud data
        results: Inference results dictionary
    """
    print("Inference Results:")
    print(f"  Processed {points.shape[0]} points")
    print(f"  Generated {results['tokens'].shape[1]} tokens")
    
    # Count obstacles
    pred_classes = results['predicted_classes'].squeeze()
    num_obstacles = (pred_classes == 1).sum().item()
    print(f"  Detected {num_obstacles} obstacles")
    
    # Show class distribution
    unique, counts = np.unique(pred_classes.numpy(), return_counts=True)
    print("  Class distribution:")
    for cls, count in zip(unique, counts):
        class_name = "Obstacle" if cls == 1 else "Background"
        print(f"    {class_name}: {count} tokens")


def main():
    """Main function to run inference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference with LIDAR Vision Transformer")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--input_data", type=str, required=True,
                        help="Path to input LIDAR data file")
    parser.add_argument("--data_format", type=str, default="kitti",
                        choices=["kitti", "npy"],
                        help="Format of input data (kitti or npy)")
    args = parser.parse_args()
    
    print("LIDAR Vision Transformer - Inference Script")
    print("=" * 45)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model(args.model_path, device)
    print("  Model loaded successfully")
    
    # Load LIDAR data
    print("Loading LIDAR data...")
    if args.data_format == "kitti":
        points = load_kitti_point_cloud(args.input_data)
    elif args.data_format == "npy":
        points = load_npy_point_cloud(args.input_data)
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")
    
    print(f"  Loaded {points.shape[0]} points with {points.shape[1]} features each")
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, points, device)
    print("  Inference completed successfully")
    
    # Visualize results
    visualize_results(points, results)
    
    # Save results (optional)
    # torch.save(results, "inference_results.pth")
    # print("  Results saved to inference_results.pth")


if __name__ == "__main__":
    # This is just a placeholder since we don't have actual data files
    # In practice, you would run:
    # python inference.py --model_path checkpoints/model_final.pth --input_data path/to/lidar/data.bin
    print("LIDAR Vision Transformer - Inference Script")
    print("=" * 45)
    print("This is a placeholder script. To run actual inference:")
    print("python inference.py --model_path checkpoints/model_final.pth --input_data path/to/lidar/data.bin")
    print("\nFor KITTI data:")
    print("python inference.py --model_path checkpoints/model_final.pth --input_data data/000000.bin --data_format kitti")
    print("\nFor .npy data:")
    print("python inference.py --model_path checkpoints/model_final.pth --input_data data/point_cloud.npy --data_format npy")