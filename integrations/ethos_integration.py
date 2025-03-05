#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ETHOS integration script for MEDTOK.

This script shows how to integrate MEDTOK with the ETHOS model.
ETHOS is a transformer-based model for tokenizing patient health timelines.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from utils.config import MedTokConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate MEDTOK with ETHOS model")
    
    parser.add_argument("--medtok_model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--ethos_model", type=str, default=None, 
                        help="Path to ETHOS model (if None, will download)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing data (MIMIC-III or MIMIC-IV)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--graph_dir", type=str, default=None, 
                        help="Directory containing graph files")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--tasks", type=str, default="all", 
                        help="Tasks to evaluate (mortality, readmission, drugrec)")
    parser.add_argument("--baseline", action="store_true", 
                        help="Run baseline ETHOS (without MEDTOK)")
    
    return parser.parse_args()


def load_medtok_model(model_path, device):
    """
    Load a trained MEDTOK model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
    
    Returns:
        MEDTOK model
    """
    logger.info(f"Loading MEDTOK model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', None)
    if config is None:
        logger.error("Config not found in checkpoint")
        return None
    
    # Update device
    config.device = device
    
    # Create model
    model = MedTok(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("MEDTOK model loaded")
    
    return model


def load_ethos_model(model_path=None, device="cuda"):
    """
    Load ETHOS model.
    
    Args:
        model_path: Path to model (if None, will download)
        device: Device to load model onto
        
    Returns:
        ETHOS model
    """
    logger.info("Loading ETHOS model...")
    
    try:
        # Import ETHOS module
        import ethos
        
        # Load model
        if model_path is not None:
            ethos_model = ethos.ETHOS.from_pretrained(model_path)
        else:
            ethos_model = ethos.ETHOS.from_pretrained("ethos-base")
        
        # Move to device
        ethos_model = ethos_model.to(device)
        
        return ethos_model
    
    except ImportError:
        logger.error("ETHOS module not found. Please install with 'pip install ethos-ehr'.")
        return None
    except Exception as e:
        logger.error(f"Error loading ETHOS model: {e}")
        return None


def tokenize_ethos_data_with_medtok(data, medtok_model, graph_dir):
    """
    Tokenize ETHOS data with MEDTOK.
    
    Args:
        data: DataFrame with ETHOS data
        medtok_model: MEDTOK model
        graph_dir: Directory containing graph files
        
    Returns:
        DataFrame with tokenized data
    """
    logger.info("Tokenizing ETHOS data with MEDTOK...")
    
    # Get device
    device = next(medtok_model.parameters()).device
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(medtok_model.config.text_encoder_model)
    
    # Initialize lists to store tokenized data
    tokenized_rows = []
    
    # Process each row
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Tokenizing"):
        # Get codes and descriptions
        codes = row["code"].split(";") if ";" in str(row["code"]) else [row["code"]]
        descriptions = row["description"].split(";") if ";" in str(row["description"]) else [row["description"]]
        
        # Make sure descriptions and codes have the same length
        if len(descriptions) != len(codes):
            descriptions = descriptions * len(codes) if len(descriptions) == 1 else descriptions[:len(codes)]
        
        # Tokenize each code
        code_tokens = []
        
        for code, description in zip(codes, descriptions):
            # Load graph file
            graph_file = os.path.join(graph_dir, f"{code}.json")
            
            if os.path.exists(graph_file):
                # Load graph
                with open(graph_file, "r") as f:
                    graph_data = json.load(f)
                
                # Convert to torch tensors
                import networkx as nx
                G = nx.node_link_graph(graph_data)
                
                # Extract node features
                node_features = []
                for node in G.nodes:
                    if "features" in G.nodes[node]:
                        node_features.append(G.nodes[node]["features"])
                    else:
                        # Create default features
                        node_features.append([0.0] * medtok_model.config.node_feature_dim)
                
                # Extract edge indices
                edge_index = []
                for src, dst in G.edges:
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])  # Add reverse edge for undirected graphs
                
                # Convert to torch tensors
                node_features = torch.tensor(node_features, dtype=torch.float).to(device)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
            else:
                # Create dummy graph
                node_features = torch.zeros((1, medtok_model.config.node_feature_dim), dtype=torch.float).to(device)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
            
            # Tokenize text
            encoded_text = tokenizer(
                description,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Create batch tensor for graph
            graph_batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
            
            # Tokenize
            with torch.no_grad():
                token_indices = medtok_model.tokenize(
                    encoded_text["input_ids"],
                    node_features,
                    edge_index,
                    graph_batch
                )
            
            # Add to code tokens
            code_tokens.append(token_indices[0].cpu().numpy().tolist())
        
        # Add to tokenized rows
        tokenized_row = row.copy()
        tokenized_row["medtok_tokens"] = code_tokens
        tokenized_rows.append(tokenized_row)
    
    # Convert to DataFrame
    tokenized_df = pd.DataFrame(tokenized_rows)
    
    logger.info(f"Tokenized {len(tokenized_df)} rows")
    
    return tokenized_df


def train_ethos_with_medtok(ethos_model, train_data, val_data, output_dir, medtok_embeddings=None):
    """
    Train ETHOS model with MEDTOK embeddings.
    
    Args:
        ethos_model: ETHOS model
        train_data: Training data
        val_data: Validation data
        output_dir: Output directory
        medtok_embeddings: MEDTOK embeddings (optional)
        
    Returns:
        Trained ETHOS model
    """
    logger.info("Training ETHOS model with MEDTOK embeddings...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up ETHOS training
    try:
        import ethos
        
        # Create ETHOS trainer
        trainer = ethos.trainer.ETHOSTrainer(
            model=ethos_model,
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
            code_embeddings=medtok_embeddings
        )
        
        # Train model
        trained_model = trainer.train()
        
        return trained_model
    
    except ImportError:
        logger.error("ETHOS module not found. Please install with 'pip install ethos-ehr'.")
        return None
    except Exception as e:
        logger.error(f"Error training ETHOS model: {e}")
        return None


def evaluate_ethos(ethos_model, test_data, tasks, output_dir):
    """
    Evaluate ETHOS model on test data.
    
    Args:
        ethos_model: ETHOS model
        test_data: Test data
        tasks: Tasks to evaluate
        output_dir: Output directory
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating ETHOS model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Set up ETHOS evaluation
    try:
        import ethos
        
        # Create ETHOS evaluator
        evaluator = ethos.evaluation.ETHOSEvaluator(
            model=ethos_model,
            test_data=test_data,
            output_dir=output_dir
        )
        
        # Evaluate for each task
        for task in tasks:
            logger.info(f"Evaluating task: {task}")
            
            # Evaluate model
            task_results = evaluator.evaluate(task=task)
            
            # Add to results
            results[task] = task_results
        
        # Save results
        results_file = os.path.join(output_dir, "results.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        return results
    
    except ImportError:
        logger.error("ETHOS module not found. Please install with 'pip install ethos-ehr'.")
        return None
    except Exception as e:
        logger.error(f"Error evaluating ETHOS model: {e}")
        return None


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    
    # Load MEDTOK model
    medtok_model = load_medtok_model(args.medtok_model, device)
    
    if medtok_model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Load ETHOS model
    ethos_model = load_ethos_model(args.ethos_model, device)
    
    if ethos_model is None:
        logger.error("Failed to load ETHOS model")
        return
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.data_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Determine tasks to evaluate
    tasks = []
    if args.tasks.lower() == "all":
        tasks = ["mortality", "readmission", "drugrec"]
    else:
        tasks = args.tasks.lower().split(",")
    
    # Load data
    try:
        import ethos
        
        # Load training data
        train_data = ethos.data.load_data(
            data_dir=args.data_dir,
            split="train"
        )
        
        # Load validation data
        val_data = ethos.data.load_data(
            data_dir=args.data_dir,
            split="val"
        )
        
        # Load test data
        test_data = ethos.data.load_data(
            data_dir=args.data_dir,
            split="test"
        )
        
        logger.info(f"Loaded data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    except ImportError:
        logger.error("ETHOS module not found. Please install with 'pip install ethos-ehr'.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Run baseline if requested
    if args.baseline:
        logger.info("Running baseline ETHOS (without MEDTOK)...")
        
        # Create output directory for baseline
        baseline_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Train baseline model
        baseline_model = train_ethos_with_medtok(
            ethos_model=ethos_model,
            train_data=train_data,
            val_data=val_data,
            output_dir=baseline_dir
        )
        
        if baseline_model is None:
            logger.error("Failed to train baseline ETHOS model")
        else:
            # Evaluate baseline model
            baseline_results = evaluate_ethos(
                ethos_model=baseline_model,
                test_data=test_data,
                tasks=tasks,
                output_dir=baseline_dir
            )
            
            if baseline_results is None:
                logger.error("Failed to evaluate baseline ETHOS model")
    
    # Tokenize data with MEDTOK
    train_data_tokenized = tokenize_ethos_data_with_medtok(train_data, medtok_model, graph_dir)
    val_data_tokenized = tokenize_ethos_data_with_medtok(val_data, medtok_model, graph_dir)
    test_data_tokenized = tokenize_ethos_data_with_medtok(test_data, medtok_model, graph_dir)
    
    # Get MEDTOK embeddings
    medtok_embeddings = {}
    
    for _, row in train_data_tokenized.iterrows():
        for code, tokens in zip(row["code"].split(";"), row["medtok_tokens"]):
            if code not in medtok_embeddings:
                medtok_embeddings[code] = tokens
    
    # Train ETHOS with MEDTOK
    medtok_dir = os.path.join(args.output_dir, "medtok")
    os.makedirs(medtok_dir, exist_ok=True)
    
    medtok_model = train_ethos_with_medtok(
        ethos_model=ethos_model,
        train_data=train_data_tokenized,
        val_data=val_data_tokenized,
        output_dir=medtok_dir,
        medtok_embeddings=medtok_embeddings
    )
    
    if medtok_model is None:
        logger.error("Failed to train ETHOS model with MEDTOK")
        return
    
    # Evaluate ETHOS with MEDTOK
    medtok_results = evaluate_ethos(
        ethos_model=medtok_model,
        test_data=test_data_tokenized,
        tasks=tasks,
        output_dir=medtok_dir
    )
    
    if medtok_results is None:
        logger.error("Failed to evaluate ETHOS model with MEDTOK")
        return
    
    # Compare results if baseline was run
    if args.baseline and 'baseline_results' in locals():
        comparison = {}
        
        for task in tasks:
            if task in baseline_results and task in medtok_results:
                # Get metric values
                baseline_metric = baseline_results[task]["auprc"]
                medtok_metric = medtok_results[task]["auprc"]
                
                # Calculate improvement
                abs_improvement = medtok_metric - baseline_metric
                rel_improvement = abs_improvement / baseline_metric * 100 if baseline_metric > 0 else 0
                
                # Add to comparison
                comparison[task] = {
                    "baseline": baseline_metric,
                    "medtok": medtok_metric,
                    "absolute_improvement": abs_improvement,
                    "relative_improvement": rel_improvement
                }
        
        # Save comparison
        comparison_file = os.path.join(args.output_dir, "comparison.json")
        
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {comparison_file}")
        
        # Print summary
        logger.info("Evaluation summary:")
        
        for task, metrics in comparison.items():
            logger.info(f"Task: {task}")
            logger.info(f"  Baseline: {metrics['baseline']:.4f}")
            logger.info(f"  MEDTOK: {metrics['medtok']:.4f}")
            logger.info(f"  Improvement: {metrics['absolute_improvement']:.4f} ({metrics['relative_improvement']:.2f}%)")


if __name__ == "__main__":
    main()
