#!/usr/bin/env python
"""
Integration test for the full MedTok pipeline.

This test verifies the complete workflow from data preprocessing through 
tokenization, model training, and prediction.
"""

import os
import sys
import pytest
import tempfile
import shutil
import json
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.medtok import MedTok
from model.text_encoder import TextEncoder
from model.graph_encoder import GraphEncoder
from model.vector_quantizer import VectorQuantizer
from model.token_packer import TokenPacker
from data.dataset import MedicalCodeDataset


@pytest.fixture(scope="module")
def test_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def sample_data(test_dir):
    """Create sample data for testing"""
    # Create directories
    os.makedirs(os.path.join(test_dir, "graphs"), exist_ok=True)
    
    # Create sample medical codes
    with open(os.path.join(test_dir, "codes.csv"), 'w') as f:
        f.write("code,system,type\n")
        f.write("E11.9,ICD10,diagnosis\n")
        f.write("I10,ICD10,diagnosis\n")
        f.write("R94.31,ICD10,diagnosis\n")
        f.write("J45.909,ICD10,diagnosis\n")
        f.write("Z79.4,ICD10,diagnosis\n")
    
    # Create sample descriptions
    with open(os.path.join(test_dir, "descriptions.csv"), 'w') as f:
        f.write("code,description\n")
        f.write("E11.9,Type 2 diabetes mellitus without complications\n")
        f.write("I10,Essential (primary) hypertension\n")
        f.write("R94.31,Abnormal electrocardiogram\n")
        f.write("J45.909,Unspecified asthma, uncomplicated\n")
        f.write("Z79.4,Long term (current) use of insulin\n")
    
    # Create sample graphs
    sample_graphs = {
        "E11.9": {
            "nodes": [
                {"id": "E11.9", "type": "diagnosis", "name": "Type 2 diabetes mellitus"},
                {"id": "insulin", "type": "chemical", "name": "Insulin"},
                {"id": "pancreas", "type": "anatomy", "name": "Pancreas"}
            ],
            "edges": [
                {"source": "E11.9", "target": "insulin", "type": "treats"},
                {"source": "E11.9", "target": "pancreas", "type": "affects"}
            ]
        },
        "I10": {
            "nodes": [
                {"id": "I10", "type": "diagnosis", "name": "Hypertension"},
                {"id": "heart", "type": "anatomy", "name": "Heart"},
                {"id": "metoprolol", "type": "chemical", "name": "Metoprolol"}
            ],
            "edges": [
                {"source": "I10", "target": "heart", "type": "affects"},
                {"source": "metoprolol", "target": "I10", "type": "treats"}
            ]
        },
        "R94.31": {
            "nodes": [
                {"id": "R94.31", "type": "diagnosis", "name": "Abnormal ECG"},
                {"id": "heart", "type": "anatomy", "name": "Heart"}
            ],
            "edges": [
                {"source": "R94.31", "target": "heart", "type": "affects"}
            ]
        },
        "J45.909": {
            "nodes": [
                {"id": "J45.909", "type": "diagnosis", "name": "Asthma"},
                {"id": "lung", "type": "anatomy", "name": "Lung"},
                {"id": "albuterol", "type": "chemical", "name": "Albuterol"}
            ],
            "edges": [
                {"source": "J45.909", "target": "lung", "type": "affects"},
                {"source": "albuterol", "target": "J45.909", "type": "treats"}
            ]
        },
        "Z79.4": {
            "nodes": [
                {"id": "Z79.4", "type": "diagnosis", "name": "Long term insulin use"},
                {"id": "insulin", "type": "chemical", "name": "Insulin"},
                {"id": "E11.9", "type": "diagnosis", "name": "Type 2 diabetes mellitus"}
            ],
            "edges": [
                {"source": "Z79.4", "target": "insulin", "type": "involves"},
                {"source": "Z79.4", "target": "E11.9", "type": "related_to"}
            ]
        }
    }
    
    # Save sample graphs to files
    for code, graph in sample_graphs.items():
        with open(os.path.join(test_dir, "graphs", f"{code}.json"), 'w') as f:
            json.dump(graph, f, indent=2)
    
    # Create sample dataset splits
    dataset = []
    for code in ["E11.9", "I10", "R94.31", "J45.909", "Z79.4"]:
        dataset.append({
            "code": code,
            "description_path": os.path.join(test_dir, "descriptions.csv"),
            "graph_path": os.path.join(test_dir, "graphs", f"{code}.json")
        })
    
    # Split dataset
    train_data = dataset[:3]
    val_data = dataset[3:4]
    test_data = dataset[4:]
    
    # Save splits
    with open(os.path.join(test_dir, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(test_dir, "val.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(test_dir, "test.json"), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Return paths
    return {
        "codes_path": os.path.join(test_dir, "codes.csv"),
        "descriptions_path": os.path.join(test_dir, "descriptions.csv"),
        "graphs_dir": os.path.join(test_dir, "graphs"),
        "model_path": os.path.join(test_dir, "model.pt"),
        "train_path": os.path.join(test_dir, "train.json"),
        "val_path": os.path.join(test_dir, "val.json"),
        "test_path": os.path.join(test_dir, "test.json")
    }


@pytest.mark.integration
def test_dataset_loading(sample_data):
    """Test dataset loading"""
    # Create dataset
    train_dataset = MedicalCodeDataset(
        data_file=sample_data["train_path"],
        descriptions_file=sample_data["descriptions_path"],
        transform=None
    )
    
    # Check dataset size
    assert len(train_dataset) == 3
    
    # Check dataset item
    item = train_dataset[0]
    assert 'code' in item
    assert 'description' in item
    assert 'graph' in item


@pytest.mark.integration
def test_model_creation():
    """Test model creation and forward pass"""
    # Create model components
    text_encoder = TextEncoder(
        model_name="bert-base-uncased",
        embedding_dim=128,
        pooling_type="mean"
    )
    
    graph_encoder = GraphEncoder(
        input_dim=16,
        hidden_dim=64,
        output_dim=128,
        num_layers=2
    )
    
    vector_quantizer = VectorQuantizer(
        codebook_size=100,
        embedding_dim=128,
        num_codebooks=4
    )
    
    token_packer = TokenPacker(
        embedding_dim=128,
        beta=0.5
    )
    
    # Create MedTok model
    model = MedTok(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        vector_quantizer=vector_quantizer,
        token_packer=token_packer,
        embedding_dim=128,
        learning_rate=1e-4
    )
    
    # Test forward pass
    batch = {
        'description': ["Type 2 diabetes mellitus without complications"],
        'graph': [{
            'nodes': np.random.randn(3, 16).astype(np.float32),
            'edges': np.array([[0, 1], [0, 2]]).astype(np.int64)
        }]
    }
    
    # Convert to tensors
    nodes = torch.from_numpy(batch['graph'][0]['nodes'])
    edges = torch.from_numpy(batch['graph'][0]['edges'])
    
    # Run forward pass
    tokens, losses = model(batch['description'], [(nodes, edges)])
    
    # Check outputs
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert isinstance(losses, dict)
    assert 'codebook_loss' in losses


@pytest.mark.integration
def test_tokenization():
    """Test tokenization of medical codes"""
    # Create a simple model
    text_encoder = TextEncoder(
        model_name="bert-base-uncased",
        embedding_dim=128,
        pooling_type="mean"
    )
    
    graph_encoder = GraphEncoder(
        input_dim=16,
        hidden_dim=64,
        output_dim=128,
        num_layers=2
    )
    
    vector_quantizer = VectorQuantizer(
        codebook_size=100,
        embedding_dim=128,
        num_codebooks=4
    )
    
    token_packer = TokenPacker(
        embedding_dim=128,
        beta=0.5
    )
    
    model = MedTok(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        vector_quantizer=vector_quantizer,
        token_packer=token_packer,
        embedding_dim=128,
        learning_rate=1e-4
    )
    
    # Test tokenization
    code = "E11.9"
    description = "Type 2 diabetes mellitus without complications"
    
    # Create a simple graph representation
    nodes = np.random.randn(3, 16).astype(np.float32)
    edges = np.array([[0, 1], [0, 2]]).astype(np.int64)
    graph = {
        'nodes': nodes,
        'edges': edges
    }
    
    # Tokenize
    tokens = model.tokenize(code, description, graph)
    
    # Check tokens
    assert isinstance(tokens, (list, torch.Tensor))
    assert len(tokens) > 0


@pytest.mark.integration
def test_end_to_end(sample_data):
    """Test end-to-end pipeline from data to tokens"""
    # 1. Load dataset
    train_dataset = MedicalCodeDataset(
        data_file=sample_data["train_path"],
        descriptions_file=sample_data["descriptions_path"],
        transform=None
    )
    
    # 2. Create model
    text_encoder = TextEncoder(
        model_name="bert-base-uncased",
        embedding_dim=128,
        pooling_type="mean"
    )
    
    graph_encoder = GraphEncoder(
        input_dim=16,
        hidden_dim=64,
        output_dim=128,
        num_layers=2
    )
    
    vector_quantizer = VectorQuantizer(
        codebook_size=100,
        embedding_dim=128,
        num_codebooks=4
    )
    
    token_packer = TokenPacker(
        embedding_dim=128,
        beta=0.5
    )
    
    model = MedTok(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        vector_quantizer=vector_quantizer,
        token_packer=token_packer,
        embedding_dim=128,
        learning_rate=1e-4
    )
    
    # 3. Run forward pass for a batch
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )
    
    for batch in dataloader:
        # Process batch
        batch_descriptions = batch['description']
        batch_graphs = []
        
        for graph in batch['graph']:
            # Create tensors from graph data
            if isinstance(graph, dict) and 'nodes' in graph and 'edges' in graph:
                nodes = torch.tensor(graph['nodes'], dtype=torch.float)
                edges = torch.tensor(graph['edges'], dtype=torch.long)
                batch_graphs.append((nodes, edges))
            else:
                # Handle missing or invalid graphs
                batch_graphs.append(None)
        
        # Forward pass
        tokens, losses = model(batch_descriptions, batch_graphs)
        
        # Check outputs
        assert isinstance(tokens, list)
        assert len(tokens) == len(batch_descriptions)
        assert isinstance(losses, dict)
        assert 'codebook_loss' in losses
        
        # Only process one batch for this test
        break
    
    # 4. Save and load model
    torch.save(model.state_dict(), sample_data["model_path"])
    
    # 5. Test tokenization with saved model
    new_model = MedTok(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        vector_quantizer=vector_quantizer,
        token_packer=token_packer,
        embedding_dim=128,
        learning_rate=1e-4
    )
    new_model.load_state_dict(torch.load(sample_data["model_path"]))
    new_model.eval()
    
    # 6. Tokenize a new code
    code = "Z79.4"
    description = "Long term (current) use of insulin"
    
    # Load graph
    with open(os.path.join(sample_data["graphs_dir"], f"{code}.json"), 'r') as f:
        graph_data = json.load(f)
    
    # Create graph representation
    nodes = np.random.randn(len(graph_data['nodes']), 16).astype(np.float32)
    edges = np.array([[0, 1], [0, 2]]).astype(np.int64)
    graph = {
        'nodes': nodes,
        'edges': edges
    }
    
    # Tokenize
    tokens = new_model.tokenize(code, description, graph)
    
    # Check tokens
    assert isinstance(tokens, (list, torch.Tensor))
    assert len(tokens) > 0
    
    print(f"Successfully tokenized {code} into {len(tokens)} tokens")
    
    # Test token embeddings
    token_embeddings = [new_model.get_token_embedding(token_id) for token_id in tokens]
    assert len(token_embeddings) == len(tokens)
    assert token_embeddings[0].shape[-1] == 128