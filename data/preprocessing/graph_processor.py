#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graph processor for medical code knowledge graphs.

This script extracts and processes subgraphs from biomedical knowledge graphs
for each medical code, capturing relevant relationships and dependencies.
"""

import os
import argparse
import pandas as pd
import json
import networkx as nx
import numpy as np
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
import random
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process medical code knowledge graphs")
    
    parser.add_argument("--input", type=str, required=True, 
                      help="Input CSV file with medical codes")
    parser.add_argument("--kg_path", type=str, required=True, 
                      help="Path to the biomedical knowledge graph (JSON or pickle)")
    parser.add_argument("--output_dir", type=str, required=True, 
                      help="Output directory for processed graph files")
    parser.add_argument("--code_col", type=str, default="code", 
                      help="Column name for medical codes")
    parser.add_argument("--system_col", type=str, default="system", 
                      help="Column name for code system (ICD9, SNOMED, etc.)")
    parser.add_argument("--mapping_file", type=str, default=None,
                      help="Optional JSON file with mappings from codes to KG nodes")
    parser.add_argument("--max_nodes", type=int, default=100,
                      help="Maximum number of nodes in extracted subgraphs")
    parser.add_argument("--hop_distance", type=int, default=2,
                      help="Maximum hop distance for extracting subgraphs")
    parser.add_argument("--batch_size", type=int, default=50,
                      help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, default=4,
                      help="Maximum number of worker processes")
    parser.add_argument("--node_feature_dim", type=int, default=128,
                      help="Dimension of node features")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_knowledge_graph(kg_path):
    """
    Load the biomedical knowledge graph.
    
    Args:
        kg_path: Path to the knowledge graph file
        
    Returns:
        NetworkX graph object
    """
    logger.info(f"Loading knowledge graph from {kg_path}...")
    
    if kg_path.endswith('.json'):
        with open(kg_path, 'r') as f:
            kg_data = json.load(f)
        G = nx.node_link_graph(kg_data)
    
    elif kg_path.endswith('.pkl') or kg_path.endswith('.pickle'):
        with open(kg_path, 'rb') as f:
            G = pickle.load(f)
    
    else:
        raise ValueError("Unsupported knowledge graph file format. Use JSON or pickle.")
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def load_code_mappings(mapping_file, code_column=None, kg_column=None):
    """
    Load mappings from medical codes to knowledge graph nodes.
    
    Args:
        mapping_file: Path to the mapping file
        code_column: Column name for medical codes
        kg_column: Column name for KG node IDs
        
    Returns:
        Dictionary mapping codes to KG nodes
    """
    logger.info(f"Loading code-to-KG mappings from {mapping_file}...")
    
    if mapping_file.endswith('.json'):
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    
    elif mapping_file.endswith('.csv'):
        mappings_df = pd.read_csv(mapping_file)
        
        if code_column is None or kg_column is None:
            # Try to infer column names
            for col in mappings_df.columns:
                if 'code' in col.lower():
                    code_column = col
                elif 'node' in col.lower() or 'kg' in col.lower() or 'entity' in col.lower():
                    kg_column = col
        
        if code_column is None or kg_column is None:
            raise ValueError("Could not infer code and KG column names. Please specify them.")
        
        mappings = {row[code_column]: row[kg_column] for _, row in mappings_df.iterrows()}
    
    else:
        raise ValueError("Unsupported mapping file format. Use JSON or CSV.")
    
    logger.info(f"Loaded {len(mappings)} code-to-KG mappings.")
    return mappings


def find_kg_nodes_for_code(code, code_system, G, mappings=None):
    """
    Find knowledge graph nodes corresponding to a medical code.
    
    Args:
        code: Medical code
        code_system: Code system (ICD9, SNOMED, etc.)
        G: Knowledge graph
        mappings: Optional mappings from codes to KG nodes
        
    Returns:
        List of KG node IDs
    """
    if mappings and code in mappings:
        # Use provided mapping
        nodes = mappings[code]
        if isinstance(nodes, list):
            return nodes
        else:
            return [nodes]
    
    # Try to find nodes by matching code
    matching_nodes = []
    
    # Exact match with code
    for node in G.nodes:
        node_data = G.nodes[node]
        
        # Check various possible attribute names
        for attr in ['code', 'id', 'identifier', 'concept_id']:
            if attr in node_data and node_data[attr] == code:
                matching_nodes.append(node)
                break
        
        # Check if node has system info that matches
        if 'system' in node_data and node_data['system'].lower() == code_system.lower() and code in str(node):
            matching_nodes.append(node)
    
    # If no exact matches, try fuzzy matching
    if not matching_nodes:
        for node in G.nodes:
            node_data = G.nodes[node]
            
            # Check if code is substring of node id or name
            if (isinstance(node, str) and code in node) or \
               ('name' in node_data and isinstance(node_data['name'], str) and code in node_data['name']):
                matching_nodes.append(node)
    
    return matching_nodes


def extract_subgraph(G, source_nodes, max_nodes=100, hop_distance=2):
    """
    Extract a subgraph around source nodes.
    
    Args:
        G: Knowledge graph
        source_nodes: Source nodes to extract subgraph from
        max_nodes: Maximum number of nodes in the subgraph
        hop_distance: Maximum hop distance from source nodes
        
    Returns:
        NetworkX subgraph
    """
    if not source_nodes:
        # Create a simple default graph if no source nodes found
        default_graph = nx.Graph()
        default_graph.add_node(0, name="default", system="default")
        return default_graph
    
    # Create a set of nodes to include in the subgraph
    subgraph_nodes = set(source_nodes)
    frontier = set(source_nodes)
    
    # Expand by BFS until we reach max_nodes or max hop distance
    for hop in range(hop_distance):
        if len(subgraph_nodes) >= max_nodes:
            break
        
        new_frontier = set()
        for node in frontier:
            if node not in G:
                continue
            
            # Add neighbors
            neighbors = list(G.neighbors(node))
            
            # Prioritize important neighbors
            neighbor_scores = []
            for neighbor in neighbors:
                if neighbor in subgraph_nodes:
                    continue
                
                # Score based on degree and edge weight
                score = G.degree(neighbor)
                if G.has_edge(node, neighbor) and 'weight' in G[node][neighbor]:
                    score *= G[node][neighbor]['weight']
                
                neighbor_scores.append((neighbor, score))
            
            # Sort by score and add top neighbors
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            for neighbor, _ in neighbor_scores[:min(5, len(neighbor_scores))]:
                new_frontier.add(neighbor)
                subgraph_nodes.add(neighbor)
                
                if len(subgraph_nodes) >= max_nodes:
                    break
            
            if len(subgraph_nodes) >= max_nodes:
                break
        
        frontier = new_frontier
        if not frontier:
            break
    
    # Extract the subgraph
    return G.subgraph(subgraph_nodes)


def generate_node_features(G, node_feature_dim=128, random_seed=42):
    """
    Generate features for nodes in the graph.
    
    Args:
        G: NetworkX graph
        node_feature_dim: Dimension of node features
        random_seed: Random seed for reproducibility
        
    Returns:
        Graph with added node features
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Create a copy of the graph
    G_with_features = G.copy()
    
    # Generate random features for nodes based on their properties
    for node in G_with_features.nodes():
        node_data = G_with_features.nodes[node]
        
        # Initialize feature vector
        feature_vector = np.zeros(node_feature_dim)
        
        # Set different parts of the feature vector based on node properties
        
        # Use degree and centrality to influence some dimensions
        degree = G.degree(node)
        degree_influence = min(degree / 10, 1.0)  # Normalize
        feature_vector[:16] = np.random.normal(degree_influence, 0.1, 16)
        
        # Use node type or system to influence other dimensions
        if 'type' in node_data:
            node_type = node_data['type']
            if isinstance(node_type, str):
                # Hash the node type to get a consistent number
                type_hash = hash(node_type) % 1000
                type_influence = (type_hash / 1000) * 2 - 1  # Between -1 and 1
                feature_vector[16:32] = np.random.normal(type_influence, 0.1, 16)
        
        elif 'system' in node_data:
            system = node_data['system']
            if isinstance(system, str):
                system_hash = hash(system) % 1000
                system_influence = (system_hash / 1000) * 2 - 1
                feature_vector[16:32] = np.random.normal(system_influence, 0.1, 16)
        
        # Use node name to influence more dimensions if available
        if 'name' in node_data:
            name = node_data['name']
            if isinstance(name, str):
                name_influence = min(len(name) / 50, 1.0)  # Normalize
                feature_vector[32:48] = np.random.normal(name_influence, 0.1, 16)
        
        # Fill the rest with random values
        feature_vector[48:] = np.random.normal(0, 0.1, node_feature_dim - 48)
        
        # Normalize to unit length
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
        
        # Add to node data
        node_data['features'] = feature_vector.tolist()
    
    return G_with_features


def process_batch(batch_data, G, mappings, args):
    """
    Process a batch of medical codes.
    
    Args:
        batch_data: List of (index, row) tuples
        G: Knowledge graph
        mappings: Code-to-KG mappings
        args: Command line arguments
        
    Returns:
        List of (code, graph) tuples
    """
    results = []
    
    for idx, row in batch_data:
        try:
            code = row[args.code_col]
            code_system = row[args.system_col] if args.system_col in row and row[args.system_col] else "UNKNOWN"
            
            # Find KG nodes for the code
            source_nodes = find_kg_nodes_for_code(code, code_system, G, mappings)
            
            # Extract subgraph
            subgraph = extract_subgraph(
                G,
                source_nodes,
                max_nodes=args.max_nodes,
                hop_distance=args.hop_distance
            )
            
            # Generate node features
            subgraph_with_features = generate_node_features(
                subgraph,
                node_feature_dim=args.node_feature_dim,
                random_seed=args.random_seed + idx  # Use different seed for each graph
            )
            
            # Convert to node-link format for serialization
            graph_data = nx.node_link_data(subgraph_with_features)
            
            results.append((code, graph_data))
        
        except Exception as e:
            logger.error(f"Error processing code {row[args.code_col]}: {e}")
    
    return results


def main():
    """Main processing function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} does not exist.")
        return
    
    # Check if knowledge graph file exists
    if not os.path.exists(args.kg_path):
        logger.error(f"Knowledge graph file {args.kg_path} does not exist.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load input data
    logger.info(f"Loading data from {args.input}...")
    try:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
        elif args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            logger.error("Unsupported input file format. Use CSV or JSON.")
            return
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    # Check if required columns exist
    if args.code_col not in df.columns:
        logger.error(f"Input file must contain column: {args.code_col}")
        return
    
    # Add system column if it doesn't exist
    if args.system_col not in df.columns:
        logger.warning(f"System column {args.system_col} not found. Adding with default value 'UNKNOWN'.")
        df[args.system_col] = "UNKNOWN"
    
    # Load knowledge graph
    try:
        G = load_knowledge_graph(args.kg_path)
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {e}")
        return
    
    # Load mappings if provided
    mappings = None
    if args.mapping_file:
        try:
            mappings = load_code_mappings(args.mapping_file)
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            return
    
    # Process data in batches with parallel execution
    logger.info(f"Processing {len(df)} medical codes...")
    
    # Convert DataFrame to list of (index, row) tuples
    data_tuples = list(df.iterrows())
    
    # Split data into batches
    batches = [data_tuples[i:i+args.batch_size] for i in range(0, len(data_tuples), args.batch_size)]
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_batch, batch, G, mappings, args)
            for batch in batches
        ]
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_results = future.result()
                
                # Save graphs to output directory
                for code, graph_data in batch_results:
                    output_file = os.path.join(args.output_dir, f"{code}.json")
                    with open(output_file, 'w') as f:
                        json.dump(graph_data, f)
            
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
    
    logger.info(f"Processing completed. Subgraphs saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
