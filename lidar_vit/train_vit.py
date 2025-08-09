"""
Standalone training script for LIDAR Vision Transformer obstacle detection model.
This script handles dataset loading, preprocessing, training loop, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from typing import Tuple, Dict, List
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lidar_vit'))

# Import our modules
from data.preprocessing import LIDARToViTTokenizer, preprocess_lidar_point_cloud
from models.vit_model import LIDARVisionTransformer
from configs.preprocessing_config import TRAINING_CONFIG, MODEL_CONFIG


class LIDARObstacleDataset(Dataset):
    """
    Dataset class for LIDAR obstacle detection.
    Generates synthetic data for demonstration purposes.
    In a real implementation, this would load actual LIDAR data and labels.
    """
    
    def __init__(self, num_samples: int = 1000, num_points: int = 50000):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            num_points: Number of points per sample
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.tokenizer = LIDARToViTTokenizer()
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tokens: Tensor of shape (N, 4) with LIDAR tokens
            coords: Tensor of shape (N, 3) with voxel coordinates
            labels: Tensor of shape (N,) with class labels
            bboxes: Tensor of shape (N, 7) with bounding box parameters
        """
        # Generate synthetic point cloud data
        points = np.random.rand(self.num_points, 4).astype(np.float32)
        points[:, 0] = points[:, 0] * 20 - 10  # x: -10 to 10 meters
        points[:, 1] = points[:, 1] * 50       # y: 0 to 50 meters
        points[:, 2] = points[:, 2] * 8 - 3    # z: -3 to 5 meters
        points[:, 3] = points[:, 3]            # intensity: 0 to 1
        
        # Preprocess point cloud to tokens
        tokens, coords = self.tokenizer.forward(points)
        
        # Generate synthetic labels (0 for background, 1 for obstacle)
        num_tokens = tokens.shape[0]
        labels = torch.zeros(num_tokens, dtype=torch.long)
        
        # Randomly assign some tokens as obstacles (10% for demonstration)
        obstacle_indices = np.random.choice(num_tokens, size=max(1, num_tokens // 10), replace=False)
        labels[obstacle_indices] = 1
        
        # Generate synthetic bounding box parameters
        bboxes = torch.randn(num_tokens, 7)  # x, y, z, w, h, l, yaw
        
        return tokens, coords, labels, bboxes


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        tokens_batch: Padded tensor of shape (B, max_N, 4)
        coords_batch: Padded tensor of shape (B, max_N, 3)
        labels_batch: Padded tensor of shape (B, max_N)
        bboxes_batch: Padded tensor of shape (B, max_N, 7)
    """
    # Get maximum sequence length in the batch
    max_len = max([tokens.shape[0] for tokens, _, _, _ in batch])
    
    # Pad sequences to the same length
    tokens_list = []
    coords_list = []
    labels_list = []
    bboxes_list = []
    
    for tokens, coords, labels, bboxes in batch:
        # Pad tokens
        num_tokens = tokens.shape[0]
        pad_tokens = torch.zeros(max_len - num_tokens, tokens.shape[1])
        padded_tokens = torch.cat([tokens, pad_tokens], dim=0)
        tokens_list.append(padded_tokens)
        
        # Pad coordinates
        pad_coords = torch.zeros(max_len - num_tokens, coords.shape[1])
        padded_coords = torch.cat([coords, pad_coords], dim=0)
        coords_list.append(padded_coords)
        
        # Pad labels
        pad_labels = torch.zeros(max_len - num_tokens, dtype=torch.long)
        padded_labels = torch.cat([labels, pad_labels], dim=0)
        labels_list.append(padded_labels)
        
        # Pad bounding boxes
        pad_bboxes = torch.zeros(max_len - num_tokens, bboxes.shape[1])
        padded_bboxes = torch.cat([bboxes, pad_bboxes], dim=0)
        bboxes_list.append(padded_bboxes)
    
    # Stack into batch tensors
    tokens_batch = torch.stack(tokens_list, dim=0)
    coords_batch = torch.stack(coords_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)
    bboxes_batch = torch.stack(bboxes_list, dim=0)
    
    return tokens_batch, coords_batch, labels_batch, bboxes_batch


def compute_metrics(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute accuracy, precision, and recall metrics.
    
    Args:
        pred_labels: Predicted labels of shape (B, N)
        true_labels: True labels of shape (B, N)
        num_classes: Number of classes
        
    Returns:
        metrics: Dictionary containing accuracy, precision, and recall
    """
    # Flatten tensors
    pred_flat = pred_labels.view(-1)
    true_flat = true_labels.view(-1)
    
    # Compute accuracy
    correct = (pred_flat == true_flat).sum().item()
    total = pred_flat.numel()
    accuracy = correct / total if total > 0 else 0.0
    
    # Compute precision and recall for obstacle class (class 1)
    true_positives = ((pred_flat == 1) & (true_flat == 1)).sum().item()
    false_positives = ((pred_flat == 1) & (true_flat == 0)).sum().item()
    false_negatives = ((pred_flat == 0) & (true_flat == 1)).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = TRAINING_CONFIG["num_epochs"],
    learning_rate: float = TRAINING_CONFIG["learning_rate"],
    weight_decay: float = TRAINING_CONFIG["weight_decay"],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir: str = "checkpoints"
) -> None:
    """
    Train the LIDAR Vision Transformer model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to use for training
        checkpoint_dir: Directory to save model checkpoints
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss functions
    classification_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for tokens_batch, coords_batch, labels_batch, bboxes_batch in progress_bar:
            # Move data to device
            tokens_batch = tokens_batch.to(device)
            coords_batch = coords_batch.to(device)
            labels_batch = labels_batch.to(device)
            bboxes_batch = bboxes_batch.to(device)
            
            # Forward pass
            class_logits, bbox_preds = model(tokens_batch, coords_batch)
            
            # Compute losses
            # Reshape for classification loss
            class_logits_flat = class_logits.view(-1, class_logits.shape[-1])
            labels_flat = labels_batch.view(-1)
            classification_loss = classification_criterion(class_logits_flat, labels_flat)
            
            # Reshape for bbox loss (only for obstacle tokens)
            obstacle_mask = (labels_batch == 1)
            if obstacle_mask.sum() > 0:
                bbox_preds_masked = bbox_preds[obstacle_mask]
                bboxes_batch_masked = bboxes_batch[obstacle_mask]
                bbox_loss = bbox_criterion(bbox_preds_masked, bboxes_batch_masked)
            else:
                bbox_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = classification_loss + 0.1 * bbox_loss  # Weight bbox loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Compute metrics
            pred_labels = torch.argmax(class_logits, dim=-1)
            batch_metrics = compute_metrics(pred_labels, labels_batch)
            
            # Update running averages
            train_loss += total_loss.item()
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Acc": f"{batch_metrics['accuracy']:.4f}",
                "Prec": f"{batch_metrics['precision']:.4f}",
                "Rec": f"{batch_metrics['recall']:.4f}"
            })
        
        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        
        with torch.no_grad():
            for tokens_batch, coords_batch, labels_batch, bboxes_batch in val_loader:
                # Move data to device
                tokens_batch = tokens_batch.to(device)
                coords_batch = coords_batch.to(device)
                labels_batch = labels_batch.to(device)
                bboxes_batch = bboxes_batch.to(device)
                
                # Forward pass
                class_logits, bbox_preds = model(tokens_batch, coords_batch)
                
                # Compute losses
                class_logits_flat = class_logits.view(-1, class_logits.shape[-1])
                labels_flat = labels_batch.view(-1)
                classification_loss = classification_criterion(class_logits_flat, labels_flat)
                
                obstacle_mask = (labels_batch == 1)
                if obstacle_mask.sum() > 0:
                    bbox_preds_masked = bbox_preds[obstacle_mask]
                    bboxes_batch_masked = bboxes_batch[obstacle_mask]
                    bbox_loss = bbox_criterion(bbox_preds_masked, bboxes_batch_masked)
                else:
                    bbox_loss = torch.tensor(0.0, device=device)
                
                total_loss = classification_loss + 0.1 * bbox_loss
                
                # Compute metrics
                pred_labels = torch.argmax(class_logits, dim=-1)
                batch_metrics = compute_metrics(pred_labels, labels_batch)
                
                # Update running averages
                val_loss += total_loss.item()
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        # Average validation metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")


def main():
    """Main function to run the training script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train LIDAR Vision Transformer for obstacle detection")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG["batch_size"],
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=TRAINING_CONFIG["weight_decay"],
                        help="Weight decay")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_train_samples", type=int, default=800,
                        help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=200,
                        help="Number of validation samples")
    args = parser.parse_args()
    
    print("LIDAR Vision Transformer - Training Script")
    print("=" * 45)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LIDARObstacleDataset(num_samples=args.num_train_samples)
    val_dataset = LIDARObstacleDataset(num_samples=args.num_val_samples)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    
    # Create model
    print("Creating model...")
    model = LIDARVisionTransformer(
        embedding_dim=MODEL_CONFIG["embedding_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        mlp_ratio=MODEL_CONFIG["mlp_ratio"],
        dropout=MODEL_CONFIG["dropout"],
        num_classes=2,  # Background and obstacle
        feature_dim=4   # x, y, z, intensity
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()