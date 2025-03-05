#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility script for tokenizing medical codes with MEDTOK.

This script provides functions for tokenizing medical codes using a trained MEDTOK model,
extracting and visualizing token embeddings, and comparing tokenizations across different codes.
"""

import os
import sys
import json
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import argparse
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from utils.config import MedTokConfig


def load_medtok_model(model_path, device="cuda"):
    """
    Load a trained MEDTOK model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
    
    Returns:
        MEDTOK model and its configuration
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError("Config not found in checkpoint")
    
    # Update device
    config.device = device
    
    # Create model
    model = MedTok(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, config


def load_graph(graph_file, node_feature_dim=128):
    """
    Load a graph file for a medical code.
    
    Args:
        graph_file: Path to the graph file
        node_feature_dim: Dimension of node features
    
    Returns:
        Tuple of (node_features, edge_index)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if file exists
    if not os.path.exists(graph_file):
        print(f"Graph file {graph_file} not found, using dummy graph")
        # Create dummy graph
        node_features = torch.zeros((1, node_feature_dim), dtype=torch.float).to(device)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        return node_features, edge_index
    
    # Load graph
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
    
    # Convert to networkx graph
    G = nx.node_link_graph(graph_data)
    
    # Extract node features
    node_features = []
    for node in G.nodes:
        if "features" in G.nodes[node]:
            node_features.append(G.nodes[node]["features"])
        else:
            # Create default features if not available
            node_features.append([0.0] * node_feature_dim)
    
    # Extract edge indices
    edge_index = []
    for src, dst in G.edges:
        edge_index.append([src, dst])
        edge_index.append([dst, src])  # Add reverse edge for undirected graphs
    
    # Convert to torch tensors
    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    return node_features, edge_index


def tokenize_medical_code(model, code_id, description, graph_file):
    """
    Tokenize a single medical code.
    
    Args:
        model: MEDTOK model
        code_id: Medical code ID
        description: Description of the medical code
        graph_file: Path to the graph file
    
    Returns:
        Token indices for the medical code
    """
    device = next(model.parameters()).device
    
    # Load text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    
    # Tokenize text
    encoded_text = tokenizer(
        description,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Load graph
    node_features, edge_index = load_graph(
        graph_file, 
        node_feature_dim=model.config.node_feature_dim
    )
    
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
    
    return token_indices


def get_token_embedding(model, token_indices):
    """
    Get token embeddings for provided token indices.
    
    Args:
        model: MEDTOK model
        token_indices: Indices of tokens in the codebook
    
    Returns:
        Token embeddings
    """
    return model.get_token_embedding(token_indices)


def visualize_tokens(model, tokens_list, code_ids, output_file=None):
    """
    Visualize token embeddings for multiple codes.
    
    Args:
        model: MEDTOK model
        tokens_list: List of token indices for different codes
        code_ids: List of code IDs corresponding to the tokens
        output_file: Path to save the visualization
    """
    # Get token embeddings
    embeddings_list = [model.get_token_embedding(tokens).mean(dim=1) for tokens in tokens_list]
    
    # Combine embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
    
    # Apply dimensionality reduction
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Keep track of starting point for each code
    offset = 0
    
    # Define colors and markers
    colors = plt.cm.rainbow(np.linspace(0, 1, len(code_ids)))
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'x', '+', '|']
    
    # Plot each code's tokens
    for i, (code, tokens) in enumerate(zip(code_ids, tokens_list)):
        n_tokens = tokens.size(1)
        plt.scatter(
            embeddings_2d[offset:offset+n_tokens, 0],
            embeddings_2d[offset:offset+n_tokens, 1],
            color=colors[i],
            marker=markers[i % len(markers)],
            alpha=0.7,
            label=f'Code: {code}'
        )
        offset += n_tokens
    
    plt.title('Token Embeddings Visualization')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


def compare_codes(model, code_ids, descriptions, graph_files, output_file=None):
    """
    Compare tokenization of multiple medical codes.
    
    Args:
        model: MEDTOK model
        code_ids: List of medical code IDs
        descriptions: List of descriptions corresponding to the codes
        graph_files: List of graph file paths corresponding to the codes
        output_file: Path to save the comparison report
    
    Returns:
        DataFrame with token comparison
    """
    # Tokenize each code
    tokens_list = []
    
    for code_id, description, graph_file in zip(code_ids, descriptions, graph_files):
        tokens = tokenize_medical_code(model, code_id, description, graph_file)
        tokens_list.append(tokens)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'code_id': code_ids,
        'description': descriptions,
        'num_tokens': [tokens.size(1) for tokens in tokens_list],
        'tokens': [tokens[0].cpu().numpy().tolist() for tokens in tokens_list]
    })
    
    # Save to file if specified
    if output_file:
        comparison.to_csv(output_file, index=False)
        print(f"Comparison saved to {output_file}")
    
    # Visualize tokens
    visualize_tokens(
        model,
        tokens_list,
        code_ids,
        output_file=output_file.replace('.csv', '.png') if output_file else None
    )
    
    return comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tokenize medical codes with MEDTOK")
    
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--code", type=str, default=None, 
                        help="Medical code ID to tokenize")
    parser.add_argument("--description", type=str, default=None, 
                        help="Description of the medical code")
    parser.add_argument("--graph", type=str, default=None, 
                        help="Path to the graph file for the code")
    parser.add_argument("--input_file", type=str, default=None, 
                        help="CSV file containing codes, descriptions, and graph files")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to save the output")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize token embeddings")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare multiple codes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check arguments
    if not args.input_file and (not args.code or not args.description):
        print("Either provide --input_file or --code and --description")
        return
    
    # Load model
    model, config = load_medtok_model(args.model, args.device)
    print(f"Model loaded from {args.model}")
    
    # Process single code
    if args.code and args.description:
        print(f"Tokenizing medical code: {args.code}")
        
        # Use default graph file if not specified
        graph_file = args.graph or f"data/processed/graphs/{args.code}.json"
        
        # Tokenize code
        tokens = tokenize_medical_code(model, args.code, args.description, graph_file)
        
        # Print token indices
        print(f"Token indices: {tokens[0].cpu().numpy()}")
        
        # Get token embeddings
        if args.visualize:
            token_embeddings = get_token_embedding(tokens)
            print(f"Token embedding shape: {token_embeddings.shape}")
            
            # Visualize token embeddings
            visualize_tokens(
                model,
                [tokens],
                [args.code],
                output_file=args.output_file if args.output_file else None
            )
    
    # Process multiple codes from input file
    elif args.input_file:
        print(f"Processing codes from {args.input_file}")
        
        # Load input file
        data = pd.read_csv(args.input_file)
        
        # Check required columns
        required_columns = ['code_id', 'description']
        if not all(col in data.columns for col in required_columns):
            print(f"Input file must contain columns: {', '.join(required_columns)}")
            return
        
        # Get graph files
        if 'graph_file' not in data.columns:
            print("Graph file column not found, using default paths")
            data['graph_file'] = data['code_id'].apply(
                lambda code: f"data/processed/graphs/{code}.json"
            )
        
        # Compare codes
        if args.compare:
            comparison = compare_codes(
                model,
                data['code_id'].tolist(),
                data['description'].tolist(),
                data['graph_file'].tolist(),
                output_file=args.output_file
            )
            print(comparison)
        
        # Tokenize each code
        else:
            results = []
            
            for _, row in data.iterrows():
                code_id = row['code_id']
                description = row['description']
                graph_file = row['graph_file']
                
                print(f"Tokenizing medical code: {code_id}")
                
                # Tokenize code
                tokens = tokenize_medical_code(model, code_id, description, graph_file)
                
                # Add to results
                results.append({
                    'code_id': code_id,
                    'description': description,
                    'tokens': tokens[0].cpu().numpy().tolist()
                })
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Save to file if specified
            if args.output_file:
                results_df.to_csv(args.output_file, index=False)
                print(f"Results saved to {args.output_file}")
            
            # Print summary
            print(f"Tokenized {len(results)} codes")


if __name__ == "__main__":
    main()
