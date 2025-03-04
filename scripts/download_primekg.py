#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download PrimeKG biomedical knowledge graph.

This script downloads the PrimeKG dataset from Harvard Dataverse
and processes it for use with MEDTOK.
"""

import os
import argparse
import requests
import zipfile
import shutil
import pandas as pd
import json
import networkx as nx
from tqdm import tqdm
import logging
import time
import urllib.request

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download PrimeKG biomedical knowledge graph")
    
    parser.add_argument("--output", type=str, required=True, 
                        help="Output directory for downloaded files")
    parser.add_argument("--skip_download", action="store_true", 
                        help="Skip download if files already exist")
    parser.add_argument("--convert_to_networkx", action="store_true", 
                        help="Convert to NetworkX format")
    parser.add_argument("--create_subgraphs", action="store_true", 
                        help="Create subgraphs for each node type")
    parser.add_argument("--max_workers", type=int, default=4, 
                        help="Maximum number of worker processes")
    
    return parser.parse_args()


def download_file(url, output_file, desc=None):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_file: Path to save the file
        desc: Description for the progress bar
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_file, output_dir):
    """
    Extract a ZIP file.
    
    Args:
        zip_file: Path to the ZIP file
        output_dir: Directory to extract to
    """
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(output_dir)
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {zip_file}: {e}")
        return False


def download_primekg(output_dir, skip_download=False):
    """
    Download PrimeKG dataset from Harvard Dataverse.
    
    Args:
        output_dir: Output directory
        skip_download: Skip download if files already exist
    """
    logger.info("Downloading PrimeKG dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs for PrimeKG files
    primekg_url = "https://dataverse.harvard.edu/api/access/dataset/:persistentId?persistentId=doi:10.7910/DVN/IXA7BM"
    
    # Download zip file
    zip_file = os.path.join(output_dir, "primekg.zip")
    extract_dir = os.path.join(output_dir, "extract")
    
    # Skip download if file exists and skip_download is True
    if os.path.exists(zip_file) and skip_download:
        logger.info(f"Skipping download, {zip_file} already exists")
    else:
        download_file(primekg_url, zip_file, "Downloading PrimeKG dataset")
    
    # Extract zip file
    if os.path.exists(extract_dir) and skip_download:
        logger.info(f"Skipping extraction, {extract_dir} already exists")
    else:
        os.makedirs(extract_dir, exist_ok=True)
        extract_zip(zip_file, extract_dir)
    
    logger.info("PrimeKG dataset downloaded and extracted")
    
    return extract_dir


def build_networkx_graph(extract_dir, output_dir):
    """
    Build a NetworkX graph from PrimeKG files.
    
    Args:
        extract_dir: Directory with extracted PrimeKG files
        output_dir: Output directory for the graph
    """
    logger.info("Building NetworkX graph from PrimeKG...")
    
    # Find data files
    nodes_file = os.path.join(extract_dir, "nodes.tsv")
    edges_file = os.path.join(extract_dir, "edges.tsv")
    
    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        logger.error(f"Could not find nodes.tsv or edges.tsv in {extract_dir}")
        return None
    
    # Load node data
    logger.info("Loading node data...")
    nodes_df = pd.read_csv(nodes_file, sep='\t')
    
    # Load edge data
    logger.info("Loading edge data...")
    edges_df = pd.read_csv(edges_file, sep='\t')
    
    # Create graph
    logger.info("Creating NetworkX graph...")
    G = nx.Graph()
    
    # Add nodes
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Adding nodes"):
        node_id = row['node_id']
        
        # Create node attributes
        node_attrs = {col: row[col] for col in nodes_df.columns if col != 'node_id'}
        
        # Add node
        G.add_node(node_id, **node_attrs)
    
    # Add edges
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Adding edges"):
        source = row['subject_id']
        target = row['object_id']
        
        # Create edge attributes
        edge_attrs = {col: row[col] for col in edges_df.columns if col not in ['subject_id', 'object_id']}
        
        # Add edge
        G.add_edge(source, target, **edge_attrs)
    
    # Save graph
    logger.info("Saving NetworkX graph...")
    graph_file = os.path.join(output_dir, "primekg.pkl")
    nx.write_gpickle(G, graph_file)
    
    # Also save as node-link format for easier inspection
    node_link_file = os.path.join(output_dir, "primekg_node_link.json")
    with open(node_link_file, 'w') as f:
        json.dump(nx.node_link_data(G), f)
    
    logger.info(f"NetworkX graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G


def create_node_type_subgraphs(G, output_dir, max_workers=4):
    """
    Create subgraphs for each node type.
    
    Args:
        G: NetworkX graph
        output_dir: Output directory for subgraphs
        max_workers: Maximum number of worker processes
    """
    logger.info("Creating subgraphs for each node type...")
    
    # Create directory for subgraphs
    subgraphs_dir = os.path.join(output_dir, "subgraphs")
    os.makedirs(subgraphs_dir, exist_ok=True)
    
    # Get node types
    node_types = set()
    for node, attrs in G.nodes(data=True):
        if 'node_type' in attrs:
            node_types.add(attrs['node_type'])
    
    logger.info(f"Found {len(node_types)} node types")
    
    # Create subgraphs for each node type
    for node_type in tqdm(node_types, desc="Creating subgraphs"):
        # Get nodes of this type
        nodes = [node for node, attrs in G.nodes(data=True) 
                if 'node_type' in attrs and attrs['node_type'] == node_type]
        
        # Create subgraph
        subgraph = G.subgraph(nodes)
        
        # Save subgraph
        subgraph_file = os.path.join(subgraphs_dir, f"{node_type}.pkl")
        nx.write_gpickle(subgraph, subgraph_file)
        
        # Also save as node-link format
        node_link_file = os.path.join(subgraphs_dir, f"{node_type}_node_link.json")
        with open(node_link_file, 'w') as f:
            json.dump(nx.node_link_data(subgraph), f)
        
        logger.info(f"Created subgraph for {node_type} with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    
    logger.info("Subgraphs created for all node types")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Download PrimeKG
    extract_dir = download_primekg(args.output, args.skip_download)
    
    # Build NetworkX graph
    if args.convert_to_networkx:
        G = build_networkx_graph(extract_dir, args.output)
        
        # Create subgraphs
        if args.create_subgraphs and G is not None:
            create_node_type_subgraphs(G, args.output, args.max_workers)
    
    logger.info("PrimeKG download and processing complete")


if __name__ == "__main__":
    main()
