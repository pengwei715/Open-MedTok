import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class MedTokConfig:
    """Configuration class for MEDTOK model."""
    
    # Dimensions and sizes
    codebook_size: int = 12000  # Default size (N) as mentioned in the paper
    embedding_dim: int = 64  # Dimension (d) for token embeddings
    text_encoder_dim: int = 768  # Output dimension from text encoder (dt)
    graph_encoder_dim: int = 300  # Output dimension from graph encoder (dg)
    num_top_k_tokens: int = 5  # Number of top K tokens to select during quantization
    
    # Codebook region proportions (what percentage of codebook is allocated to each part)
    text_specific_ratio: float = 0.3
    graph_specific_ratio: float = 0.3
    shared_ratio: float = 0.4  # Remainder is allocated to shared tokens
    
    # Model hyperparameters
    alpha: float = 0.25  # Weighing factor for VQ objective
    beta: float = 0.2  # Parameter for shared information regularization
    lambda_val: float = 0.1  # Parameter for specific information regularization
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_steps: int = 3000  # As mentioned in the paper
    dropout_rate: float = 0.1
    
    # Other settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Text encoder settings
    text_encoder_model: str = "bert-base-uncased"  # Default text encoder
    text_encoder_trainable: bool = False  # Freeze text encoder during training as per paper
    
    # Graph settings
    max_num_nodes: int = 100  # Maximum number of nodes in a subgraph
    max_num_edges: int = 300  # Maximum number of edges in a subgraph
    node_feature_dim: int = 128  # Dimension of node features in the graph
    
    def __post_init__(self):
        """Validate and compute derived parameters after initialization."""
        # Calculate actual sizes for different regions of the codebook
        self.text_specific_size = int(self.codebook_size * self.text_specific_ratio)
        self.graph_specific_size = int(self.codebook_size * self.graph_specific_ratio)
        self.shared_size = self.codebook_size - self.text_specific_size - self.graph_specific_size
        
        # Create directory structure if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
