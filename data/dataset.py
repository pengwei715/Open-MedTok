import os
import torch
import json
import pandas as pd
import networkx as nx
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer


class MedicalCodeDataset(Dataset):
    """
    Dataset class for medical codes with text descriptions and graph structures.
    
    This dataset loads medical codes, their descriptions, and associated knowledge subgraphs
    for training the MEDTOK model.
    """
    
    def __init__(self, data_dir, config, split="train", transform=None):
        """
        Initialize the medical code dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            config: Configuration object
            split: Data split ("train", "val", or "test")
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load the code systems data
        self.codes_df = pd.read_csv(os.path.join(data_dir, f"medical_codes_{split}.csv"))
        
        # Load the text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
        
        # Process the data
        self._process_data()
    
    def _process_data(self):
        """Process the raw data into suitable format for the model."""
        self.codes = []
        self.descriptions = []
        self.graph_data = []
        
        for idx, row in self.codes_df.iterrows():
            code = row["code"]
            description = row["description"]
            graph_file = os.path.join(self.data_dir, "graphs", f"{code}.json")
            
            # Skip if graph file doesn't exist
            if not os.path.exists(graph_file):
                continue
            
            # Load the graph data
            with open(graph_file, "r") as f:
                graph_json = json.load(f)
            
            # Convert to networkx graph
            G = nx.node_link_graph(graph_json)
            
            # Prepare node features (assuming they're already in the graph)
            node_features = []
            for node in G.nodes():
                if "features" in G.nodes[node]:
                    node_features.append(G.nodes[node]["features"])
                else:
                    # Create default features if not available
                    node_features.append([0.0] * self.config.node_feature_dim)
            
            # Convert to PyTorch Geometric Data object
            edge_index = []
            for src, dst in G.edges():
                edge_index.append([src, dst])
                edge_index.append([dst, src])  # Add reverse edge for undirected graphs
            
            if not edge_index:  # Skip isolated nodes
                continue
            
            # Convert to torch tensors
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Create PyG Data object
            graph_data = Data(x=node_features, edge_index=edge_index)
            
            # Add to the dataset
            self.codes.append(code)
            self.descriptions.append(description)
            self.graph_data.append(graph_data)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.codes)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing code, description, and graph data
        """
        code = self.codes[idx]
        description = self.descriptions[idx]
        graph_data = self.graph_data[idx]
        
        # Tokenize the description
        encoded_text = self.tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)
        
        sample = {
            "code": code,
            "text": description,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_features": graph_data.x,
            "graph_edge_index": graph_data.edge_index
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class MedicalCodePairDataset(Dataset):
    """
    Dataset class for pairs of medical codes to train token packing.
    
    This dataset creates pairs of related medical codes to help the model
    learn shared and specific information across modalities.
    """
    
    def __init__(self, data_dir, config, split="train", transform=None):
        """
        Initialize the medical code pair dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            config: Configuration object
            split: Data split ("train", "val", or "test")
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load the code systems data
        self.codes_df = pd.read_csv(os.path.join(data_dir, f"medical_codes_{split}.csv"))
        
        # Load the code relations data
        self.relations_df = pd.read_csv(os.path.join(data_dir, "code_relations.csv"))
        
        # Load the text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
        
        # Process the data
        self._process_data()
    
    def _process_data(self):
        """Process the raw data into suitable pairs for the model."""
        # Create a code to index mapping
        self.code_to_idx = {code: i for i, code in enumerate(self.codes_df["code"])}
        
        # Create pairs of related codes
        self.pairs = []
        for _, row in self.relations_df.iterrows():
            code1 = row["code1"]
            code2 = row["code2"]
            relation_type = row["relation_type"]
            
            # Skip if either code is not in the dataset
            if code1 not in self.code_to_idx or code2 not in self.code_to_idx:
                continue
            
            idx1 = self.code_to_idx[code1]
            idx2 = self.code_to_idx[code2]
            
            self.pairs.append((idx1, idx2, relation_type))
    
    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair of samples from the dataset.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Dictionary containing data for both medical codes
        """
        idx1, idx2, relation_type = self.pairs[idx]
        
        # Get the data for the first code
        row1 = self.codes_df.iloc[idx1]
        code1 = row1["code"]
        description1 = row1["description"]
        graph_file1 = os.path.join(self.data_dir, "graphs", f"{code1}.json")
        
        # Get the data for the second code
        row2 = self.codes_df.iloc[idx2]
        code2 = row2["code"]
        description2 = row2["description"]
        graph_file2 = os.path.join(self.data_dir, "graphs", f"{code2}.json")
        
        # Load the graph data for the first code
        with open(graph_file1, "r") as f:
            graph_json1 = json.load(f)
        
        # Load the graph data for the second code
        with open(graph_file2, "r") as f:
            graph_json2 = json.load(f)
        
        # Convert to networkx graphs
        G1 = nx.node_link_graph(graph_json1)
        G2 = nx.node_link_graph(graph_json2)
        
        # Prepare node features
        node_features1 = []
        for node in G1.nodes():
            if "features" in G1.nodes[node]:
                node_features1.append(G1.nodes[node]["features"])
            else:
                node_features1.append([0.0] * self.config.node_feature_dim)
        
        node_features2 = []
        for node in G2.nodes():
            if "features" in G2.nodes[node]:
                node_features2.append(G2.nodes[node]["features"])
            else:
                node_features2.append([0.0] * self.config.node_feature_dim)
        
        # Convert to edge indices
        edge_index1 = []
        for src, dst in G1.edges():
            edge_index1.append([src, dst])
            edge_index1.append([dst, src])
        
        edge_index2 = []
        for src, dst in G2.edges():
            edge_index2.append([src, dst])
            edge_index2.append([dst, src])
        
        # Convert to torch tensors
        node_features1 = torch.tensor(node_features1, dtype=torch.float)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long).t().contiguous()
        
        node_features2 = torch.tensor(node_features2, dtype=torch.float)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long).t().contiguous()
        
        # Tokenize the descriptions
        encoded_text1 = self.tokenizer(
            description1,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        encoded_text2 = self.tokenizer(
            description2,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids1 = encoded_text1["input_ids"].squeeze(0)
        attention_mask1 = encoded_text1["attention_mask"].squeeze(0)
        
        input_ids2 = encoded_text2["input_ids"].squeeze(0)
        attention_mask2 = encoded_text2["attention_mask"].squeeze(0)
        
        sample = {
            "code1": code1,
            "text1": description1,
            "input_ids1": input_ids1,
            "attention_mask1": attention_mask1,
            "graph_features1": node_features1,
            "graph_edge_index1": edge_index1,
            
            "code2": code2,
            "text2": description2,
            "input_ids2": input_ids2,
            "attention_mask2": attention_mask2,
            "graph_features2": node_features2,
            "graph_edge_index2": edge_index2,
            
            "relation_type": relation_type
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
