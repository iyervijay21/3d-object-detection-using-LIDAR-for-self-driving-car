"""
Configuration file for LIDAR Vision Transformer preprocessing.
"""

# Default preprocessing parameters
DEFAULT_PREPROCESSING_CONFIG = {
    # Voxel grid parameters
    "grid_size": [100, 100, 20],      # Number of voxels in x, y, z dimensions
    "voxel_size": [0.1, 0.1, 0.1],   # Size of each voxel in meters
    "max_points_per_voxel": 32,      # Maximum points to store per voxel
    
    # Feature processing
    "normalize_coords": True,        # Normalize coordinates to [0, 1]
    "use_intensity": True,           # Include intensity values in features
    
    # Performance
    "batch_size": 1,                 # Batch size for processing
}

# Model input dimensions
MODEL_CONFIG = {
    "embedding_dim": 256,            # Dimension of token embeddings
    "num_heads": 8,                  # Number of attention heads
    "num_layers": 6,                 # Number of transformer layers
    "mlp_ratio": 4,                  # MLP hidden dim ratio to embedding dim
    "dropout": 0.1,                  # Dropout rate
}

# Training parameters
TRAINING_CONFIG = {
    "learning_rate": 1e-4,           # Learning rate
    "weight_decay": 1e-4,            # Weight decay
    "batch_size": 8,                 # Training batch size
    "num_epochs": 100,               # Number of training epochs
}