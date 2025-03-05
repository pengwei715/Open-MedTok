#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for MEDTOK components.

This script contains unit tests for all components of the MEDTOK
tokenizer, including text encoder, graph encoder, vector quantizer,
and token packer.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np
import json
import networkx as nx
from tempfile import TemporaryDirectory

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from model.text_encoder import TextEncoder, WeightedPoolingTextEncoder
from model.graph_encoder import GraphEncoder, GATGraphEncoder, HierarchicalGraphEncoder
from model.vector_quantizer import VectorQuantizer, MultiPartCodebook
from model.token_packer import TokenPacker
from utils.config import MedTokConfig
from utils.metrics import compute_reconstruction_error, compute_codebook_utilization


class MockData:
    """Mock data for testing."""
    
    @staticmethod
    def create_config():
        """Create a config for testing."""
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
    
    @staticmethod
    def create_text_data(batch_size=4):
        """Create mock text data."""
        # Mock input_ids and attention_mask
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask
    
    @staticmethod
    def create_graph_data(batch_size=4, num_nodes=10, feature_dim=32):
        """Create mock graph data."""
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
    
    @staticmethod
    def create_mock_graph_file():
        """Create a mock graph file for testing."""
        G = nx.Graph()
        
        # Add nodes
        for i in range(5):
            G.add_node(i, name=f"node_{i}")
        
        # Add edges
        for i in range(4):
            G.add_edge(i, i+1, weight=1.0)
        
        # Add node features
        for i in range(5):
            G.nodes[i]["features"] = [float(j) for j in range(32)]
        
        # Convert to node-link format
        graph_data = nx.node_link_data(G)
        
        # Save to temporary file
        with TemporaryDirectory() as tmpdir:
            graph_file = os.path.join(tmpdir, "test_graph.json")
            with open(graph_file, "w") as f:
                json.dump(graph_data, f)
            
            return graph_file


class TestTextEncoder(unittest.TestCase):
    """Test cases for TextEncoder."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MockData.create_config()
    
    def test_text_encoder_initialization(self):
        """Test TextEncoder initialization."""
        encoder = TextEncoder(self.config)
        self.assertIsInstance(encoder, TextEncoder)
        self.assertEqual(encoder.transformer.config.hidden_size, 768)
    
    def test_weighted_pooling_text_encoder_initialization(self):
        """Test WeightedPoolingTextEncoder initialization."""
        encoder = WeightedPoolingTextEncoder(self.config)
        self.assertIsInstance(encoder, WeightedPoolingTextEncoder)
        self.assertEqual(encoder.transformer.config.hidden_size, 768)
    
    def test_text_encoder_forward_shape(self):
        """Test TextEncoder forward pass shape."""
        encoder = TextEncoder(self.config)
        
        # Mock input
        input_ids, attention_mask = MockData.create_text_data()
        
        # Forward pass
        with torch.no_grad():
            output = encoder(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (input_ids.size(0), self.config.embedding_dim))


class TestGraphEncoder(unittest.TestCase):
    """Test cases for GraphEncoder."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MockData.create_config()
    
    def test_graph_encoder_initialization(self):
        """Test GraphEncoder initialization."""
        encoder = GraphEncoder(self.config)
        self.assertIsInstance(encoder, GraphEncoder)
    
    def test_gat_graph_encoder_initialization(self):
        """Test GATGraphEncoder initialization."""
        encoder = GATGraphEncoder(self.config)
        self.assertIsInstance(encoder, GATGraphEncoder)
    
    def test_hierarchical_graph_encoder_initialization(self):
        """Test HierarchicalGraphEncoder initialization."""
        encoder = HierarchicalGraphEncoder(self.config)
        self.assertIsInstance(encoder, HierarchicalGraphEncoder)
    
    def test_graph_encoder_forward_shape(self):
        """Test GraphEncoder forward pass shape."""
        encoder = GraphEncoder(self.config)
        
        # Mock input
        node_features, edge_index, batch = MockData.create_graph_data(
            batch_size=4,
            num_nodes=20,
            feature_dim=self.config.node_feature_dim
        )
        
        # Forward pass
        with torch.no_grad():
            output = encoder(node_features, edge_index, batch)
        
        # Check output shape (batch_size, embedding_dim)
        self.assertEqual(output.shape, (4, self.config.embedding_dim))


class TestVectorQuantizer(unittest.TestCase):
    """Test cases for VectorQuantizer."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MockData.create_config()
    
    def test_vector_quantizer_initialization(self):
        """Test VectorQuantizer initialization."""
        vq = VectorQuantizer(self.config)
        self.assertIsInstance(vq, VectorQuantizer)
        self.assertEqual(vq.codebook.weight.shape, (self.config.codebook_size, self.config.embedding_dim))
    
    def test_vector_quantizer_forward(self):
        """Test VectorQuantizer forward pass."""
        vq = VectorQuantizer(self.config)
        
        # Mock input
        z = torch.randn(4, self.config.embedding_dim)
        
        # Forward pass
        with torch.no_grad():
            z_q, indices, q_loss = vq(z)
        
        # Check output shapes
        self.assertEqual(z_q.shape, z.shape)
        self.assertEqual(indices.shape, (4, self.config.num_top_k_tokens))
        self.assertIsInstance(q_loss, torch.Tensor)
        self.assertEqual(q_loss.dim(), 0)  # Scalar
    
    def test_multi_part_codebook_initialization(self):
        """Test MultiPartCodebook initialization."""
        codebook = MultiPartCodebook(self.config)
        self.assertIsInstance(codebook, MultiPartCodebook)
        
        # Check region sizes
        self.assertEqual(codebook.text_specific_end_idx - codebook.text_specific_start_idx, 
                         int(self.config.codebook_size * self.config.text_specific_ratio))
        
        self.assertEqual(codebook.graph_specific_end_idx - codebook.graph_specific_start_idx, 
                         int(self.config.codebook_size * self.config.graph_specific_ratio))
        
        self.assertEqual(codebook.shared_end_idx - codebook.shared_start_idx, 
                         self.config.codebook_size - 
                         int(self.config.codebook_size * self.config.text_specific_ratio) - 
                         int(self.config.codebook_size * self.config.graph_specific_ratio))


class TestTokenPacker(unittest.TestCase):
    """Test cases for TokenPacker."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MockData.create_config()
    
    def test_token_packer_initialization(self):
        """Test TokenPacker initialization."""
        packer = TokenPacker(self.config)
        self.assertIsInstance(packer, TokenPacker)
    
    def test_token_packer_forward(self):
        """Test TokenPacker forward pass."""
        packer = TokenPacker(self.config)
        
        # Mock inputs
        batch_size = 4
        dim = self.config.embedding_dim
        
        text_embedding = torch.randn(batch_size, dim)
        graph_embedding = torch.randn(batch_size, dim)
        e_text_s = torch.randn(batch_size, dim)
        e_graph_s = torch.randn(batch_size, dim)
        e_text_c = torch.randn(batch_size, dim)
        e_graph_c = torch.randn(batch_size, dim)
        
        # Forward pass
        with torch.no_grad():
            result = packer(text_embedding, graph_embedding, e_text_s, e_graph_s, e_text_c, e_graph_c)
        
        # Check output
        self.assertIn('token_packing_loss', result)
        self.assertIn('common_loss', result)
        self.assertIn('specific_loss', result)
        self.assertIn('kl_loss', result)
        
        # Check loss dimensions
        self.assertEqual(result['token_packing_loss'].dim(), 0)  # Scalar
        self.assertEqual(result['common_loss'].dim(), 0)  # Scalar
        self.assertEqual(result['specific_loss'].dim(), 0)  # Scalar
        self.assertEqual(result['kl_loss'].dim(), 0)  # Scalar


class TestMedTok(unittest.TestCase):
    """Test cases for MedTok."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MockData.create_config()
    
    def test_medtok_initialization(self):
        """Test MedTok initialization."""
        model = MedTok(self.config)
        self.assertIsInstance(model, MedTok)
        
        # Check components
        self.assertIsInstance(model.text_encoder, nn.Module)
        self.assertIsInstance(model.graph_encoder, nn.Module)
        self.assertIsInstance(model.codebook, MultiPartCodebook)
        self.assertIsInstance(model.token_packer, TokenPacker)
    
    def test_medtok_forward(self):
        """Test MedTok forward pass."""
        model = MedTok(self.config)
        
        # Mock inputs
        input_ids, attention_mask = MockData.create_text_data()
        node_features, edge_index, batch = MockData.create_graph_data(
            batch_size=4,
            num_nodes=20,
            feature_dim=self.config.node_feature_dim
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, node_features, edge_index, batch)
        
        # Check outputs
        self.assertIn('token_indices', outputs)
        self.assertIn('z_text_s', outputs)
        self.assertIn('z_graph_s', outputs)
        self.assertIn('z_text_c', outputs)
        self.assertIn('z_graph_c', outputs)
        self.assertIn('e_text_s', outputs)
        self.assertIn('e_graph_s', outputs)
        self.assertIn('e_text_c', outputs)
        self.assertIn('e_graph_c', outputs)
        self.assertIn('quantization_loss', outputs)
        self.assertIn('packing_loss', outputs)
        self.assertIn('total_loss', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['token_indices'].shape[0], 4)  # Batch size
        
        # Check that the token indices are the right shape
        expected_tokens = 4 * self.config.num_top_k_tokens  # 4 different token types
        self.assertEqual(outputs['token_indices'].shape[1], expected_tokens)
    
    def test_medtok_tokenize(self):
        """Test MedTok tokenize method."""
        model = MedTok(self.config)
        
        # Mock inputs
        input_ids, attention_mask = MockData.create_text_data(batch_size=1)
        node_features, edge_index, batch = MockData.create_graph_data(
            batch_size=1,
            num_nodes=5,
            feature_dim=self.config.node_feature_dim
        )
        
        # Tokenize
        with torch.no_grad():
            token_indices = model.tokenize(input_ids, node_features, edge_index, batch)
        
        # Check output
        self.assertEqual(token_indices.shape[0], 1)  # Batch size
        
        # Check that the token indices are the right shape
        expected_tokens = 4 * self.config.num_top_k_tokens  # 4 different token types
        self.assertEqual(token_indices.shape[1], expected_tokens)
    
    def test_medtok_get_token_embedding(self):
        """Test MedTok get_token_embedding method."""
        model = MedTok(self.config)
        
        # Mock token indices
        token_indices = torch.randint(0, self.config.codebook_size, (1, 4 * self.config.num_top_k_tokens))
        
        # Get token embeddings
        with torch.no_grad():
            token_embeddings = model.get_token_embedding(token_indices)
        
        # Check output shape
        self.assertEqual(token_embeddings.shape[0], 1)  # Batch size
        self.assertEqual(token_embeddings.shape[1], 4 * self.config.num_top_k_tokens)  # Number of tokens
        self.assertEqual(token_embeddings.shape[2], self.config.embedding_dim)  # Embedding dimension


class TestMetrics(unittest.TestCase):
    """Test cases for metrics functions."""
    
    def test_compute_reconstruction_error(self):
        """Test compute_reconstruction_error function."""
        # Mock inputs
        original = torch.randn(4, 32)
        reconstructed = original + 0.1 * torch.randn(4, 32)  # Add some noise
        
        # Compute error
        error = compute_reconstruction_error(original, reconstructed)
        
        # Check output
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0.0)
    
    def test_compute_codebook_utilization(self):
        """Test compute_codebook_utilization function."""
        # Mock inputs
        token_indices = np.random.randint(0, 100, 1000)
        codebook_size = 100
        
        # Compute utilization
        utilization = compute_codebook_utilization(token_indices, codebook_size)
        
        # Check output
        self.assertIsInstance(utilization, float)
        self.assertGreaterEqual(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)


if __name__ == '__main__':
    unittest.main()
