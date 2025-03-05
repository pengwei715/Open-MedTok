#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process medical codes and link them with knowledge graph representations.

This script processes medical code datasets, enhances their descriptions,
and extracts relevant subgraphs from a biomedical knowledge graph.
"""

import os
import sys
import argparse
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
import logging
import re
from concurrent.futures import ProcessPoolExecutor
import torch
from transformers import AutoTokenizer, AutoModel

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing.text_processor import clean_text, standardize_description, enrich_description
from data.preprocessing.graph_processor import extract_subgraph, generate_node_features, find_kg_nodes_for_code

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process medical codes and link with knowledge graph")
    
    parser.add_argument("--codes_dir", type=str, required=True, 
                        help="Directory containing medical code files")
    parser.add_argument("--kg_dir", type=str, required=True, 
                        help="Directory containing knowledge graph files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for processed data")
    parser.add_argument("--code_file", type=str, default="all_medical_codes.csv", 
                        help="Filename of the medical codes CSV in codes_dir")
    parser.add_argument("--kg_file", type=str, default="primekg.pkl", 
                        help="Filename of the knowledge graph in kg_dir")
    parser.add_argument("--mapping_file", type=str, default=None, 
                        help="Optional file mapping medical codes to KG nodes")
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, default=4, 
                        help="Maximum number of worker processes")
    parser.add_argument("--node_feature_dim", type=int, default=128, 
                        help="Dimension of node features")
    parser.add_argument("--max_nodes", type=int, default=100, 
                        help="Maximum number of nodes in extracted subgraphs")
    parser.add_argument("--hop_distance", type=int, default=2, 
                        help="Maximum hop distance for subgraph extraction")
    parser.add_argument("--enhance_descriptions", action="store_true", 
                        help="Enhance code descriptions using an LLM")
    parser.add_argument("--skip_existing", action="store_true", 
                        help="Skip processing codes that already have graph files")
    parser.add_argument("--train_val_test_split", type=str, default="0.7,0.1,0.2", 
                        help="Comma-separated train, validation, test split proportions")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_knowledge_graph(kg_dir, kg_file):
    """
    Load the biomedical knowledge graph.
    
    Args:
        kg_dir: Directory containing the knowledge graph
        kg_file: Filename of the knowledge graph
    
    Returns:
        NetworkX graph
    """
    logger.info(f"Loading knowledge graph from {os.path.join(kg_dir, kg_file)}...")
    
    kg_path = os.path.join(kg_dir, kg_file)
    
    if not os.path.exists(kg_path):
        logger.error(f"Knowledge graph file not found: {kg_path}")
        return None
    
    # Load based on file extension
    if kg_file.endswith('.pkl') or kg_file.endswith('.pickle'):
        G = nx.read_gpickle(kg_path)
    elif kg_file.endswith('.json'):
        with open(kg_path, 'r') as f:
            graph_data = json.load(f)
        G = nx.node_link_graph(graph_data)
    else:
        logger.error(f"Unsupported knowledge graph format: {kg_file}")
        return None
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def load_code_mappings(mapping_file):
    """
    Load mappings from medical codes to knowledge graph nodes.
    
    Args:
        mapping_file: Path to the mapping file
    
    Returns:
        Dictionary mapping codes to node IDs
    """
    if not mapping_file or not os.path.exists(mapping_file):
        logger.info("No mapping file provided or file does not exist")
        return {}
    
    logger.info(f"Loading code-to-KG mappings from {mapping_file}...")
    
    if mapping_file.endswith('.json'):
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    elif mapping_file.endswith('.csv'):
        df = pd.read_csv(mapping_file)
        
        # Try to infer column names
        code_col = None
        kg_col = None
        
        for col in df.columns:
            if 'code' in col.lower():
                code_col = col
            elif 'node' in col.lower() or 'kg' in col.lower() or 'entity' in col.lower():
                kg_col = col
        
        if not code_col or not kg_col:
            logger.error("Could not infer code and KG column names")
            return {}
        
        mappings = {row[code_col]: row[kg_col] for _, row in df.iterrows()}
    else:
        logger.error(f"Unsupported mapping file format: {mapping_file}")
        return {}
    
    logger.info(f"Loaded {len(mappings)} code-to-KG mappings")
    return mappings


def load_medical_codes(codes_dir, code_file):
    """
    Load medical codes dataset.
    
    Args:
        codes_dir: Directory containing the medical codes
        code_file: Filename of the medical codes CSV
    
    Returns:
        DataFrame with medical codes
    """
    logger.info(f"Loading medical codes from {os.path.join(codes_dir, code_file)}...")
    
    code_path = os.path.join(codes_dir, code_file)
    
    if not os.path.exists(code_path):
        logger.error(f"Medical codes file not found: {code_path}")
        return None
    
    # Load based on file extension
    if code_file.endswith('.csv'):
        df = pd.read_csv(code_path)
    elif code_file.endswith('.json'):
        with open(code_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        logger.error(f"Unsupported medical codes format: {code_file}")
        return None
    
    # Ensure required columns exist
    required_columns = ['code', 'description', 'system']
    if not all(col in df.columns for col in required_columns):
        # Try to adapt to different column names
        rename_map = {}
        
        for col in df.columns:
            if 'code' in col.lower() and 'code' not in df.columns:
                rename_map[col] = 'code'
            elif 'desc' in col.lower() and 'description' not in df.columns:
                rename_map[col] = 'description'
            elif 'system' in col.lower() or 'type' in col.lower() and 'system' not in df.columns:
                rename_map[col] = 'system'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Check again
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Medical codes file is missing required columns: {', '.join(missing)}")
            return None
    
    logger.info(f"Loaded {len(df)} medical codes")
    return df


def enhance_code_descriptions(codes_df):
    """
    Enhance medical code descriptions to provide more context.
    
    Args:
        codes_df: DataFrame with medical codes
    
    Returns:
        DataFrame with enhanced descriptions
    """
    logger.info("Enhancing medical code descriptions...")
    
    enhanced_df = codes_df.copy()
    
    # Apply text processing functions
    enhanced_df['description'] = enhanced_df.apply(
        lambda row: standardize_description(
            row, 'code', 'description', 'system'
        ),
        axis=1
    )
    
    enhanced_df['description'] = enhanced_df.apply(
        lambda row: enrich_description(
            row, 'code', 'description', 'system'
        ),
        axis=1
    )
    
    return enhanced_df


def enhance_descriptions_with_model(codes_df, batch_size=32):
    """
    Enhance medical code descriptions using a language model.
    
    Args:
        codes_df: DataFrame with medical codes
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with enhanced descriptions
    """
    logger.info("Enhancing descriptions with a language model...")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Function to get embedding
    def get_embedding(text):
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    # Function to find similar descriptions
    def find_similar_descriptions(description, system, embeddings, descriptions, systems, top_k=5):
        if not description:
            return ""
        
        # Get embedding for the query
        query_embedding = get_embedding(description)
        
        # Filter by system
        mask = systems == system
        filtered_embeddings = embeddings[mask]
        filtered_descriptions = descriptions[mask]
        
        if len(filtered_embeddings) == 0:
            return description
        
        # Calculate similarities
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k similar descriptions
        top_indices = np.argsort(similarities)[-top_k:]
        top_descriptions = filtered_descriptions[top_indices]
        
        # Combine descriptions
        enhanced = description
        for desc in top_descriptions:
            if desc != description and len(desc) > len(description):
                enhanced = f"{enhanced} {desc}"
                break
        
        return enhanced
    
    # Process in batches
    enhanced_df = codes_df.copy()
    unique_systems = enhanced_df['system'].unique()
    
    # Process each system separately
    for system in unique_systems:
        logger.info(f"Processing system: {system}")
        
        # Filter by system
        system_df = enhanced_df[enhanced_df['system'] == system]
        
        # Skip if too few codes
        if len(system_df) < 10:
            logger.info(f"Skipping system {system} with only {len(system_df)} codes")
            continue
        
        # Compute embeddings
        descriptions = []
        embeddings = []
        
        for i in range(0, len(system_df), batch_size):
            batch = system_df.iloc[i:i+batch_size]
            batch_descriptions = batch['description'].tolist()
            
            for desc in batch_descriptions:
                descriptions.append(desc)
                embeddings.append(get_embedding(desc))
        
        # Convert to numpy arrays
        descriptions = np.array(descriptions)
        embeddings = np.array(embeddings)
        systems = np.array([system] * len(descriptions))
        
        # Find similar descriptions
        for idx, row in tqdm(
            enhanced_df[enhanced_df['system'] == system].iterrows(),
            total=len(enhanced_df[enhanced_df['system'] == system]),
            desc=f"Enhancing {system} descriptions"
        ):
            enhanced_desc = find_similar_descriptions(
                row['description'],
                system,
                embeddings,
                descriptions,
                systems
            )
            
            enhanced_df.at[idx, 'description'] = enhanced_desc
    
    return enhanced_df


def process_code_batch(batch_data, G, code_mappings, output_graphs_dir, args):
    """
    Process a batch of medical codes.
    
    Args:
        batch_data: List of (code, description, system) tuples
        G: NetworkX graph of the knowledge graph
        code_mappings: Dictionary mapping codes to KG nodes
        output_graphs_dir: Directory to save extracted subgraphs
        args: Command line arguments
    
    Returns:
        List of processed codes with updated attributes
    """
    results = []
    
    for code, description, system in batch_data:
        try:
            # Skip if output file already exists and skip_existing is True
            output_file = os.path.join(output_graphs_dir, f"{code}.json")
            if args.skip_existing and os.path.exists(output_file):
                # Add to results without processing
                results.append({
                    'code': code,
                    'description': description,
                    'system': system,
                    'has_graph': True
                })
                continue
            
            # Find knowledge graph nodes for the code
            kg_nodes = find_kg_nodes_for_code(code, system, G, code_mappings)
            
            # Extract subgraph
            subgraph = extract_subgraph(
                G,
                kg_nodes,
                max_nodes=args.max_nodes,
                hop_distance=args.hop_distance
            )
            
            # Generate node features
            subgraph_with_features = generate_node_features(
                subgraph,
                node_feature_dim=args.node_feature_dim,
                random_seed=args.seed
            )
            
            # Convert to node-link format for serialization
            graph_data = nx.node_link_data(subgraph_with_features)
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(graph_data, f)
            
            # Add to results
            results.append({
                'code': code,
                'description': description,
                'system': system,
                'has_graph': True
            })
        
        except Exception as e:
            logger.error(f"Error processing code {code}: {e}")
            
            # Add to results with error flag
            results.append({
                'code': code,
                'description': description,
                'system': system,
                'has_graph': False,
                'error': str(e)
            })
    
    return results


def create_train_val_test_split(df, split_proportions, seed=42):
    """
    Create train, validation, and test splits of the data.
    
    Args:
        df: DataFrame to split
        split_proportions: List of [train, val, test] proportions
        seed: Random seed
    
    Returns:
        DataFrame with added 'split' column
    """
    # Parse split proportions
    train_prop, val_prop, test_prop = split_proportions
    assert abs(sum(split_proportions) - 1.0) < 1e-6, "Split proportions must sum to 1.0"
    
    # Copy the DataFrame
    df_split = df.copy()
    
    # Add split column
    df_split['split'] = 'train'  # Default
    
    # Sample indices for validation and test sets
    np.random.seed(seed)
    indices = np.random.permutation(len(df_split))
    
    # Calculate split sizes
    train_size = int(train_prop * len(df_split))
    val_size = int(val_prop * len(df_split))
    
    # Assign splits
    df_split.iloc[indices[train_size:train_size+val_size], df_split.columns.get_loc('split')] = 'val'
    df_split.iloc[indices[train_size+val_size:], df_split.columns.get_loc('split')] = 'test'
    
    # Log split sizes
    logger.info(f"Train: {(df_split['split'] == 'train').sum()} examples")
    logger.info(f"Validation: {(df_split['split'] == 'val').sum()} examples")
    logger.info(f"Test: {(df_split['split'] == 'test').sum()} examples")
    
    return df_split


def main():
    """Main processing function."""
    # Parse arguments
    args = parse_args()
    
    # Parse train-val-test split proportions
    split_proportions = [float(x) for x in args.train_val_test_split.split(',')]
    if len(split_proportions) != 3:
        logger.error("train_val_test_split must contain 3 comma-separated values")
        return
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    output_graphs_dir = os.path.join(args.output_dir, "graphs")
    os.makedirs(output_graphs_dir, exist_ok=True)
    
    # Load medical codes
    codes_df = load_medical_codes(args.codes_dir, args.code_file)
    if codes_df is None:
        return
    
    # Load knowledge graph
    G = load_knowledge_graph(args.kg_dir, args.kg_file)
    if G is None:
        return
    
    # Load code mappings
    code_mappings = load_code_mappings(args.mapping_file)
    
    # Enhance code descriptions
    if args.enhance_descriptions:
        try:
            # First apply basic enhancement
            codes_df = enhance_code_descriptions(codes_df)
            
            # Then try to use a language model for further enhancement
            codes_df = enhance_descriptions_with_model(codes_df)
        except Exception as e:
            logger.error(f"Error enhancing descriptions: {e}")
            logger.info("Falling back to basic description enhancement")
            codes_df = enhance_code_descriptions(codes_df)
    
    # Process codes in batches
    logger.info(f"Processing {len(codes_df)} medical codes...")
    
    # Prepare batches
    batch_data = [(row['code'], row['description'], row['system']) 
                 for _, row in codes_df.iterrows()]
    
    batches = [batch_data[i:i+args.batch_size] 
              for i in range(0, len(batch_data), args.batch_size)]
    
    # Process batches in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_code_batch, 
                batch, 
                G, 
                code_mappings,
                output_graphs_dir,
                args
            )
            for batch in batches
        ]
        
        # Collect results
        for future in tqdm(futures, total=len(batches), desc="Processing batches"):
            batch_results = future.result()
            results.extend(batch_results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Log statistics
    success_count = results_df['has_graph'].sum()
    logger.info(f"Successfully processed {success_count} out of {len(results_df)} codes")
    
    # Create train/val/test split
    results_df = create_train_val_test_split(results_df, split_proportions, args.seed)
    
    # Save processed data
    for split in ['train', 'val', 'test']:
        split_df = results_df[results_df['split'] == split]
        split_file = os.path.join(args.output_dir, f"medical_codes_{split}.csv")
        split_df.to_csv(split_file, index=False)
        logger.info(f"Saved {len(split_df)} {split} examples to {split_file}")
    
    # Save all processed data
    all_file = os.path.join(args.output_dir, "medical_codes_all.csv")
    results_df.to_csv(all_file, index=False)
    logger.info(f"Saved all {len(results_df)} processed codes to {all_file}")
    
    # Save statistics
    stats = {
        'total_codes': len(results_df),
        'successful_processing': success_count,
        'success_rate': success_count / len(results_df) if len(results_df) > 0 else 0,
        'codes_by_system': results_df.groupby('system').size().to_dict(),
        'codes_by_split': results_df.groupby('split').size().to_dict()
    }
    
    stats_file = os.path.join(args.output_dir, "processing_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_file}")
    logger.info("Processing completed")


if __name__ == "__main__":
    main()
