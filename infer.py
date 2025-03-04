import os
import argparse
import torch
import json
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Data

from model.medtok import MedTok
from utils.config import MedTokConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Use MEDTOK model for tokenization")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--input_file", type=str, required=True, help="Input file with medical codes")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save tokenized codes")
    parser.add_argument("--graph_dir", type=str, default="data/graphs", help="Directory with graph files")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
    
    return parser.parse_args()


def load_model(model_path, device):
    """
    Load a trained MEDTOK model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model onto
    
    Returns:
        Loaded model and config
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Update device
    config.device = device
    
    # Create model
    model = MedTok(config).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, config


def load_data(input_file, graph_dir):
    """
    Load data for tokenization.
    
    Args:
        input_file: Input file with medical codes
        graph_dir: Directory with graph files
    
    Returns:
        DataFrame with code data
    """
    # Load the input data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported input file format. Use CSV or JSON.")
    
    # Check if required columns exist
    required_columns = ['code', 'description']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain columns: {', '.join(required_columns)}")
    
    # Add graph file paths
    df['graph_file'] = df['code'].apply(lambda x: os.path.join(graph_dir, f"{x}.json"))
    
    # Filter out codes without graph files
    valid_codes = df['graph_file'].apply(os.path.exists)
    if not valid_codes.all():
        print(f"Warning: {(~valid_codes).sum()} codes do not have graph files and will be skipped.")
        df = df[valid_codes]
    
    return df


def prepare_batch(codes, descriptions, graph_files, tokenizer, config, device):
    """
    Prepare a batch for tokenization.
    
    Args:
        codes: List of medical codes
        descriptions: List of text descriptions
        graph_files: List of graph file paths
        tokenizer: Text tokenizer
        config: Model configuration
        device: Device to load tensors onto
    
    Returns:
        Batch data for the model
    """
    # Tokenize the descriptions
    encoded_inputs = tokenizer(
        descriptions,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    
    # Load graph data
    graph_list = []
    for graph_file in graph_files:
        with open(graph_file, 'r') as f:
            graph_json = json.load(f)
        
        # Convert to networkx graph
        G = nx.node_link_graph(graph_json)
        
        # Prepare node features
        node_features = []
        for node in G.nodes():
            if "features" in G.nodes[node]:
                node_features.append(G.nodes[node]["features"])
            else:
                # Create default features if not available
                node_features.append([0.0] * config.node_feature_dim)
        
        # Prepare edge index
        edge_index = []
        for src, dst in G.edges():
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # Add reverse edge for undirected graphs
        
        # Convert to torch tensors
        if not node_features:
            # Create a single default node if graph is empty
            node_features = [[0.0] * config.node_feature_dim]
            edge_index = [[0, 0]]  # Self-loop
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create PyG Data object
        graph_data = Data(x=node_features_tensor, edge_index=edge_index_tensor)
        graph_list.append(graph_data)
    
    # Batch the graphs
    from torch_geometric.data import Batch as GraphBatch
    batched_graphs = GraphBatch.from_data_list(graph_list)
    
    return {
        "codes": codes,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graph_features": batched_graphs.x.to(device),
        "graph_edge_index": batched_graphs.edge_index.to(device),
        "graph_batch": batched_graphs.batch.to(device)
    }


def tokenize_codes(model, data_df, config, batch_size, device):
    """
    Tokenize medical codes using the MEDTOK model.
    
    Args:
        model: Trained MEDTOK model
        data_df: DataFrame with code data
        config: Model configuration
        batch_size: Batch size for inference
        device: Device to use
    
    Returns:
        DataFrame with original data and tokenized codes
    """
    # Get tokenizer from the model
    tokenizer = model.text_encoder.tokenizer
    
    # Results storage
    tokenized_codes = []
    
    # Process in batches
    for i in tqdm(range(0, len(data_df), batch_size), desc="Tokenizing"):
        batch_df = data_df.iloc[i:i+batch_size]
        
        # Prepare batch
        batch = prepare_batch(
            batch_df['code'].tolist(),
            batch_df['description'].tolist(),
            batch_df['graph_file'].tolist(),
            tokenizer,
            config,
            device
        )
        
        # Tokenize
        with torch.no_grad():
            token_indices = model.tokenize(
                batch['input_ids'],
                batch['graph_features'],
                batch['graph_edge_index'],
                batch['graph_batch']
            )
        
        # Convert to Python lists
        token_indices_list = token_indices.cpu().numpy().tolist()
        
        # Store results
        for code, tokens in zip(batch['codes'], token_indices_list):
            tokenized_codes.append({
                'code': code,
                'tokens': tokens
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(tokenized_codes)
    
    # Merge with original data
    merged_df = pd.merge(data_df, results_df, on='code')
    
    return merged_df


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, args.device)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data_df = load_data(args.input_file, args.graph_dir)
    
    # Tokenize codes
    print("Tokenizing medical codes...")
    results_df = tokenize_codes(model, data_df, config, args.batch_size, args.device)
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    if args.output_file.endswith('.csv'):
        # Convert tokens list to string for CSV
        results_df['tokens'] = results_df['tokens'].apply(lambda x: ','.join(map(str, x)))
        results_df.to_csv(args.output_file, index=False)
    elif args.output_file.endswith('.json'):
        results_json = results_df.to_dict(orient='records')
        with open(args.output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
    else:
        raise ValueError("Unsupported output file format. Use CSV or JSON.")
    
    print(f"Tokenization completed for {len(results_df)} codes.")


if __name__ == "__main__":
    main()
