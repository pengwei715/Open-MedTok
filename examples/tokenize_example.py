#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the MEDTOK tokenizer.

This example shows how to load a trained MEDTOK model and tokenize medical codes.
"""

import os
import sys
import json
import torch
import networkx as nx
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from utils.config import MedTokConfig


def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    config = MedTokConfig()
    
    # Update config with loaded values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def load_model(model_path, config=None):
    """Load a trained MEDTOK model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config from checkpoint if not provided
    if config is None:
        config = checkpoint['config']
    
    # Update device
    config.device = str(device)
    
    # Create model
    model = MedTok(config).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, config


def tokenize_medical_code(model, code_desc, graph_path, verbose=False):
    """
    Tokenize a single medical code.
    
    Args:
        model: MEDTOK model
        code_desc: Text description of the medical code
        graph_path: Path to the graph file
        verbose: Whether to print verbose information
    
    Returns:
        Tokenized representation
    """
    device = next(model.parameters()).device
    
    # Tokenize text
    tokenizer = model.text_encoder.tokenizer
    encoded_text = tokenizer(
        code_desc,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Load graph
    with open(graph_path, 'r') as f:
        graph_json = json.load(f)
    
    G = nx.node_link_graph(graph_json)
    
    if verbose:
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Extract node features and edge index
    node_features = []
    for node in G.nodes:
        if "features" in G.nodes[node]:
            node_features.append(G.nodes[node]["features"])
        else:
            node_features.append([0.0] * model.config.node_feature_dim)
    
    edge_index = []
    for src, dst in G.edges:
        edge_index.append([src, dst])
        edge_index.append([dst, src])  # Add reverse edge for undirected graphs
    
    # Convert to torch tensors
    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    # Create batch tensor for graph
    graph_batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
    
    # Tokenize
    with torch.no_grad():
        token_indices = model.tokenize(
            encoded_text["input_ids"],
            node_features,
            edge_index,
            graph_batch
        )
    
    if verbose:
        print(f"Token indices: {token_indices}")
        
        # Get token embeddings
        token_embeddings = model.get_token_embedding(token_indices)
        print(f"Token embeddings shape: {token_embeddings.shape}")
    
    return token_indices.cpu().numpy()


def main():
    """Main function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Tokenize medical codes with MEDTOK")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--code", type=str, required=True, help="Medical code to tokenize")
    parser.add_argument("--desc", type=str, required=True, help="Description of the medical code")
    parser.add_argument("--graph", type=str, required=True, help="Path to the graph file")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load model
    model, _ = load_model(args.model, config)
    
    print(f"Model loaded successfully from {args.model}")
    
    # Tokenize the medical code
    tokens = tokenize_medical_code(model, args.desc, args.graph, args.verbose)
    
    print(f"\nMedical Code: {args.code}")
    print(f"Description: {args.desc}")
    print(f"Tokens: {tokens}")


if __name__ == "__main__":
    main()
