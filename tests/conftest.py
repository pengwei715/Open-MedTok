"""
pytest configuration for MedTok tests.

This file contains fixtures and configuration for pytest to properly run
all MedTok tests, both unit and integration tests.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.medtok import MedTok
from model.text_encoder import TextEncoder
from model.graph_encoder import GraphEncoder
from model.vector_quantizer import VectorQuantizer
from model.token_packer import TokenPacker
from utils.config import MedTokConfig


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = MedTokConfig()
    config.codebook_size = 100
    config.embedding_dim = 32
    config.text_encoder_dim = 64
    config.graph_encoder_dim = 64
    config.num_top_k_tokens = 3
    config.text_specific_ratio = 0.3
    config.graph_specific_ratio = 0.3
    config.shared_ratio = 0.4
    config.alpha = 0.25
    config.beta = 0.2
    config.lambda_val = 0.1
    config.batch_size = 4
    config.device = "cpu"
    config.text_encoder_model = "google-bert/bert-base-uncased"
    config.node_feature_dim = 32
    return config

@pytest.fixture
def mock_text_data():
    """Create mock text data."""
    batch_size = 4
    # Mock input_ids and attention_mask
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

@pytest.fixture
def mock_graph_data():
    """Create mock graph data."""
    batch_size = 4
    num_nodes = 20
    feature_dim = 32
    
    # Mock node features
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Mock edge index (fully connected)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Mock batch assignment
    nodes_per_graph = num_nodes // batch_size
    batch = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(batch_size):
        batch[i*nodes_per_graph:(i+1)*nodes_per_graph] = i
    
    return node_features, edge_index, batch