#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for evaluating MEDTOK on medical QA tasks.

This script evaluates the MEDTOK tokenizer on medical QA tasks using MedDDx dataset,
showcasing how MEDTOK can enhance LLM performance for medical question answering.
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
import re
from collections import defaultdict

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
    parser = argparse.ArgumentParser(description="Evaluate MEDTOK on medical QA tasks")
    
    parser.add_argument("--medtok_model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--llm_model", type=str, required=True, 
                        help="LLM model name (llama3.1-8b or claude-3-sonnet)")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset to evaluate (medddx)")
    parser.add_argument("--dataset_dir", type=str, default=None, 
                        help="Directory containing dataset files")
    parser.add_argument("--graph_dir", type=str, default=None, 
                        help="Directory containing graph files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation")
    parser.add_argument("--max_tokens", type=int, default=2048, 
                        help="Maximum tokens for LLM responses")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--no_baseline", action="store_true", 
                        help="Skip baseline evaluation")
    
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


def load_llm(model_name, device):
    """
    Load a language model.
    
    Args:
        model_name: Name of the model
        device: Device to load the model onto
    
    Returns:
        Language model and tokenizer
    """
    logger.info(f"Loading LLM: {model_name}...")
    
    if model_name.startswith("llama"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Get model version - try "llama3.1-8b" or similar formats
        model_version = model_name.replace("llama", "meta-llama/Llama-3")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        model = AutoModelForCausalLM.from_pretrained(
            model_version,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Return model and tokenizer
        return model, tokenizer
    
    elif model_name.startswith("claude"):
        # For Claude, we'll use the Anthropic API
        try:
            import anthropic
            
            # Check if API key is set
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
                return None, None
            
            # Create Anthropic client
            client = anthropic.Anthropic(api_key=api_key)
            
            # Return client and None for tokenizer (we'll use the client directly)
            return client, None
        
        except ImportError:
            logger.error("Anthropic Python library not found. Please install with 'pip install anthropic'.")
            return None, None
    
    else:
        logger.error(f"Unsupported LLM: {model_name}")
        return None, None


def load_medddx_dataset(dataset_dir):
    """
    Load MedDDx dataset.
    
    Args:
        dataset_dir: Directory containing MedDDx dataset
    
    Returns:
        Dictionary with MedDDx data
    """
    logger.info("Loading MedDDx dataset...")
    
    # Define the paths to the dataset files
    if dataset_dir:
        basic_file = os.path.join(dataset_dir, "basic.csv")
        intermediate_file = os.path.join(dataset_dir, "intermediate.csv")
        expert_file = os.path.join(dataset_dir, "expert.csv")
    else:
        # If dataset_dir is not provided, use default paths
        # Adjust these paths to match your environment
        data_root = os.environ.get("MEDDDX_DIR", "data/medddx")
        basic_file = os.path.join(data_root, "basic.csv")
        intermediate_file = os.path.join(data_root, "intermediate.csv")
        expert_file = os.path.join(data_root, "expert.csv")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [basic_file, intermediate_file, expert_file]):
        logger.error("MedDDx dataset files not found")
        return None
    
    # Load datasets
    basic_df = pd.read_csv(basic_file)
    intermediate_df = pd.read_csv(intermediate_file)
    expert_df = pd.read_csv(expert_file)
    
    # Add difficulty level
    basic_df["difficulty"] = "basic"
    intermediate_df["difficulty"] = "intermediate"
    expert_df["difficulty"] = "expert"
    
    # Combine datasets
    all_df = pd.concat([basic_df, intermediate_df, expert_df], ignore_index=True)
    
    # Return data
    return {
        "basic": basic_df,
        "intermediate": intermediate_df,
        "expert": expert_df,
        "all": all_df
    }


def extract_diseases_from_qa(qa_data):
    """
    Extract disease mentions from QA data.
    
    Args:
        qa_data: DataFrame with QA data
    
    Returns:
        DataFrame with disease mentions
    """
    logger.info("Extracting disease mentions from QA data...")
    
    # Initialize list to store disease mentions
    disease_mentions = []
    
    # Extract from questions
    for _, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc="Extracting diseases"):
        question = row["question"]
        options = [row["option_A"], row["option_B"], row["option_C"], row["option_D"]]
        answer = row[f"option_{row['answer']}"]
        
        # Extract disease mentions from question and options
        # For simplicity, we'll use some common disease patterns
        disease_patterns = [
            r"([\w\s]+) diagnosis",
            r"diagnose ([\w\s]+)",
            r"symptoms of ([\w\s]+)",
            r"([\w\s]+) disease",
            r"([\w\s]+) syndrome",
            r"([\w\s]+) disorder"
        ]
        
        # Extract from question
        question_diseases = []
        for pattern in disease_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            question_diseases.extend(matches)
        
        # Extract from options
        option_diseases = []
        for option in options:
            option_diseases.append(option)  # Treat each option as a potential disease
        
        # Extract from answer
        answer_disease = answer
        
        # Add to disease mentions
        disease_mentions.append({
            "question_id": row["id"],
            "question_diseases": question_diseases,
            "option_diseases": option_diseases,
            "answer_disease": answer_disease,
            "difficulty": row["difficulty"]
        })
    
    # Convert to DataFrame
    disease_df = pd.DataFrame(disease_mentions)
    
    return disease_df


def map_diseases_to_codes(disease_df, medical_codes_file):
    """
    Map disease mentions to medical codes.
    
    Args:
        disease_df: DataFrame with disease mentions
        medical_codes_file: Path to medical codes file
    
    Returns:
        DataFrame with disease-to-code mappings
    """
    logger.info("Mapping diseases to medical codes...")
    
    # Load medical codes
    if not os.path.exists(medical_codes_file):
        logger.error(f"Medical codes file not found: {medical_codes_file}")
        return None
    
    medical_codes = pd.read_csv(medical_codes_file)
    
    # Initialize list to store mappings
    mappings = []
    
    # Process each disease mention
    for _, row in tqdm(disease_df.iterrows(), total=len(disease_df), desc="Mapping to codes"):
        question_id = row["question_id"]
        
        # Process question diseases
        question_codes = []
        for disease in row["question_diseases"]:
            # Find matching codes
            matches = medical_codes[
                medical_codes["description"].str.contains(disease, case=False, na=False)
            ]
            
            if not matches.empty:
                question_codes.extend(matches["code"].tolist())
        
        # Process option diseases
        option_codes = []
        for disease in row["option_diseases"]:
            # Find matching codes
            matches = medical_codes[
                medical_codes["description"].str.contains(disease, case=False, na=False)
            ]
            
            if not matches.empty:
                option_codes.extend(matches["code"].tolist())
        
        # Process answer disease
        answer_codes = []
        disease = row["answer_disease"]
        
        # Find matching codes
        matches = medical_codes[
            medical_codes["description"].str.contains(disease, case=False, na=False)
        ]
        
        if not matches.empty:
            answer_codes.extend(matches["code"].tolist())
        
        # Add to mappings
        mappings.append({
            "question_id": question_id,
            "question_codes": question_codes,
            "option_codes": option_codes,
            "answer_codes": answer_codes,
            "difficulty": row["difficulty"]
        })
    
    # Convert to DataFrame
    mappings_df = pd.DataFrame(mappings)
    
    return mappings_df


def tokenize_codes(model, codes, descriptions, graph_dir):
    """
    Tokenize medical codes.
    
    Args:
        model: MEDTOK model
        codes: List of medical codes
        descriptions: List of code descriptions
        graph_dir: Directory containing graph files
    
    Returns:
        List of token indices
    """
    # Create device
    device = next(model.parameters()).device
    
    # Initialize list to store token indices
    all_token_indices = []
    
    # Process each code
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
                    node_features.append([0.0] * model.config.node_feature_dim)
            
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
            node_features = torch.zeros((1, model.config.node_feature_dim), dtype=torch.float).to(device)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        
        # Tokenize text
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        
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
            token_indices = model.tokenize(
                encoded_text["input_ids"],
                node_features,
                edge_index,
                graph_batch
            )
        
        # Add to result
        all_token_indices.append(token_indices[0].cpu().numpy())
    
    return all_token_indices


def generate_medtok_prefix(model, code_mappings, graph_dir):
    """
    Generate MEDTOK prefix for QA examples.
    
    Args:
        model: MEDTOK model
        code_mappings: DataFrame with code mappings
        graph_dir: Directory containing graph files
    
    Returns:
        Dictionary mapping question IDs to prefix tokens
    """
    logger.info("Generating MEDTOK prefixes...")
    
    # Initialize dictionary to store prefixes
    prefixes = {}
    
    # Process each question
    for _, row in tqdm(code_mappings.iterrows(), total=len(code_mappings), desc="Generating prefixes"):
        question_id = row["question_id"]
        
        # Combine all codes
        all_codes = (
            row["question_codes"] + 
            row["option_codes"] + 
            row["answer_codes"]
        )
        
        # Remove duplicates
        unique_codes = list(set(all_codes))
        
        if not unique_codes:
            # No codes found, skip
            prefixes[question_id] = None
            continue
        
        # Get descriptions for codes
        descriptions = [f"Medical code: {code}" for code in unique_codes]
        
        # Tokenize codes
        token_indices = tokenize_codes(model, unique_codes, descriptions, graph_dir)
        
        # Store prefix tokens
        prefixes[question_id] = token_indices
    
    return prefixes


def prepare_prompt(row, prefix_tokens=None, use_prefix=True):
    """
    Prepare prompt for the LLM.
    
    Args:
        row: DataFrame row with question data
        prefix_tokens: MEDTOK prefix tokens (optional)
        use_prefix: Whether to use prefix tokens
    
    Returns:
        Prompt string
    """
    # Extract question and options
    question = row["question"]
    options = {
        "A": row["option_A"],
        "B": row["option_B"],
        "C": row["option_C"],
        "D": row["option_D"]
    }
    
    # Construct prompt
    prompt = f"Answer the following medical question. Choose the best option (A, B, C, or D).\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    
    prompt += "\nPlease respond with the letter of the correct answer (A, B, C, or D)."
    
    # Add prefix tokens if available and requested
    if prefix_tokens is not None and use_prefix:
        prefix_info = "Context (medical codes relevant to this question):\n"
        prefix_info += f"[MEDTOK prefix tokens: {len(prefix_tokens)} token sequences identified]\n\n"
        
        prompt = prefix_info + prompt
    
    return prompt


def get_llm_predictions(model, tokenizer, prompts, device, max_tokens=2048, batch_size=8):
    """
    Get predictions from LLM.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        device: Device to use
        max_tokens: Maximum number of tokens to generate
        batch_size: Batch size
    
    Returns:
        List of predictions
    """
    logger.info(f"Getting predictions from LLM for {len(prompts)} prompts...")
    
    predictions = []
    
    # Check if model is Anthropic client
    if hasattr(model, "messages"):
        # Using Anthropic API
        for prompt in tqdm(prompts, desc="Generating responses"):
            try:
                message = model.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response = message.content[0].text
                predictions.append(response)
            
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                predictions.append("")
    
    else:
        # Using local model
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize prompts
            batch_inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
            
            # Decode responses
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract generated parts
            for j, response in enumerate(responses):
                prompt = batch_prompts[j]
                generated = response[len(prompt):]
                predictions.append(generated)
    
    return predictions


def extract_answer(response):
    """
    Extract answer from LLM response.
    
    Args:
        response: LLM response string
    
    Returns:
        Extracted answer (A, B, C, or D)
    """
    # Look for direct answer patterns
    patterns = [
        r"The correct answer is ([A-D])",
        r"Answer: ([A-D])",
        r"([A-D]) is the correct answer",
        r"Option ([A-D])",
        r"^([A-D])$"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[0].upper()
    
    # If no direct pattern matches, look for answer letters in the text
    letters = re.findall(r'\b([A-D])[:\.\)]', response)
    if letters:
        return letters[-1].upper()  # Return the last letter mentioned
    
    # If still no match, return None
    return None


def evaluate_predictions(predictions, qa_data):
    """
    Evaluate LLM predictions.
    
    Args:
        predictions: List of predictions
        qa_data: DataFrame with QA data
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating predictions...")
    
    # Initialize counters
    total = len(predictions)
    correct = 0
    extracted = 0
    
    # Calculate metrics by difficulty
    metrics_by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0, "extracted": 0})
    
    # Evaluate each prediction
    for i, (prediction, (_, row)) in enumerate(zip(predictions, qa_data.iterrows())):
        # Extract answer
        extracted_answer = extract_answer(prediction)
        correct_answer = row["answer"]
        difficulty = row["difficulty"]
        
        # Update counters
        metrics_by_difficulty[difficulty]["total"] += 1
        
        if extracted_answer:
            extracted += 1
            metrics_by_difficulty[difficulty]["extracted"] += 1
            
            if extracted_answer == correct_answer:
                correct += 1
                metrics_by_difficulty[difficulty]["correct"] += 1
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    extraction_rate = extracted / total if total > 0 else 0
    
    # Calculate accuracy by difficulty
    accuracy_by_difficulty = {}
    for difficulty, counts in metrics_by_difficulty.items():
        difficulty_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        difficulty_extraction = counts["extracted"] / counts["total"] if counts["total"] > 0 else 0
        
        accuracy_by_difficulty[difficulty] = {
            "accuracy": difficulty_accuracy,
            "extraction_rate": difficulty_extraction,
            "total": counts["total"],
            "correct": counts["correct"],
            "extracted": counts["extracted"]
        }
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "extraction_rate": extraction_rate,
        "total": total,
        "correct": correct,
        "extracted": extracted,
        "by_difficulty": accuracy_by_difficulty
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load MEDTOK model
    medtok_model = load_medtok_model(args.medtok_model, device)
    
    if medtok_model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Load LLM
    llm_model, llm_tokenizer = load_llm(args.llm_model, device)
    
    if llm_model is None:
        logger.error(f"Failed to load LLM: {args.llm_model}")
        return
    
    # Load dataset
    if args.dataset.lower() == "medddx":
        dataset = load_medddx_dataset(args.dataset_dir)
        
        if dataset is None:
            logger.error("Failed to load MedDDx dataset")
            return
        
        # Use intermediate for training and expert for testing
        train_data = dataset["intermediate"]
        test_data = dataset["expert"]
        basic_data = dataset["basic"]
        
        logger.info(f"Loaded MedDDx dataset: {len(train_data)} intermediate, {len(test_data)} expert, {len(basic_data)} basic")
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")
        return
    
    # Extract disease mentions
    disease_df = extract_diseases_from_qa(dataset["all"])
    
    # Map diseases to codes
    medical_codes_file = os.path.join(args.dataset_dir, "medical_codes.csv") if args.dataset_dir else "data/medical_codes.csv"
    code_mappings = map_diseases_to_codes(disease_df, medical_codes_file)
    
    if code_mappings is None:
        logger.error("Failed to map diseases to codes")
        return
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.dataset_dir, "graphs") if args.dataset_dir else "data/graphs"
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Generate MEDTOK prefixes
    prefixes = generate_medtok_prefix(medtok_model, code_mappings, graph_dir)
    
    # Prepare evaluation datasets
    eval_datasets = [
        ("expert", test_data),
        ("basic", basic_data)
    ]
    
    # Initialize results dictionary
    results = {}
    
    # Test on each dataset
    for dataset_name, dataset_data in eval_datasets:
        logger.info(f"Evaluating on {dataset_name} dataset...")
        
        # Prepare prompts with MEDTOK prefixes
        medtok_prompts = []
        
        for _, row in dataset_data.iterrows():
            question_id = row["id"]
            prefix_tokens = prefixes.get(question_id)
            prompt = prepare_prompt(row, prefix_tokens, use_prefix=True)
            medtok_prompts.append(prompt)
        
        # Get predictions with MEDTOK
        medtok_predictions = get_llm_predictions(
            llm_model, 
            llm_tokenizer, 
            medtok_prompts, 
            device,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size
        )
        
        # Evaluate MEDTOK predictions
        medtok_metrics = evaluate_predictions(medtok_predictions, dataset_data)
        
        # Baseline evaluation (without MEDTOK)
        if not args.no_baseline:
            # Prepare prompts without MEDTOK prefixes
            baseline_prompts = []
            
            for _, row in dataset_data.iterrows():
                prompt = prepare_prompt(row, None, use_prefix=False)
                baseline_prompts.append(prompt)
            
            # Get predictions without MEDTOK
            baseline_predictions = get_llm_predictions(
                llm_model, 
                llm_tokenizer, 
                baseline_prompts, 
                device,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size
            )
            
            # Evaluate baseline predictions
            baseline_metrics = evaluate_predictions(baseline_predictions, dataset_data)
            
            # Calculate improvement
            accuracy_improvement = medtok_metrics["accuracy"] - baseline_metrics["accuracy"]
            relative_improvement = accuracy_improvement / baseline_metrics["accuracy"] * 100 if baseline_metrics["accuracy"] > 0 else 0
            
            # Add to results
            results[dataset_name] = {
                "medtok": medtok_metrics,
                "baseline": baseline_metrics,
                "absolute_improvement": accuracy_improvement,
                "relative_improvement": relative_improvement
            }
        else:
            # Add only MEDTOK results
            results[dataset_name] = {
                "medtok": medtok_metrics
            }
        
        # Save detailed predictions
        predictions_data = []
        
        for i, (_, row) in enumerate(dataset_data.iterrows()):
            prediction_item = {
                "question_id": row["id"],
                "question": row["question"],
                "options": {
                    "A": row["option_A"],
                    "B": row["option_B"],
                    "C": row["option_C"],
                    "D": row["option_D"]
                },
                "correct_answer": row["answer"],
                "medtok_prediction": medtok_predictions[i],
                "medtok_extracted": extract_answer(medtok_predictions[i])
            }
            
            if not args.no_baseline:
                prediction_item.update({
                    "baseline_prediction": baseline_predictions[i],
                    "baseline_extracted": extract_answer(baseline_predictions[i])
                })
            
            predictions_data.append(prediction_item)
        
        # Save predictions
        predictions_file = os.path.join(args.output_dir, f"{dataset_name}_predictions.json")
        
        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f, indent=2)
        
        logger.info(f"Predictions saved to {predictions_file}")
    
    # Save overall results
    results_file = os.path.join(args.output_dir, "results.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    logger.info("Evaluation summary:")
    
    for dataset_name, dataset_results in results.items():
        logger.info(f"Dataset: {dataset_name}")
        
        if "baseline" in dataset_results:
            # Print baseline vs. MEDTOK
            logger.info(f"  MEDTOK accuracy: {dataset_results['medtok']['accuracy']:.4f}")
            logger.info(f"  Baseline accuracy: {dataset_results['baseline']['accuracy']:.4f}")
            logger.info(f"  Improvement: {dataset_results['absolute_improvement']:.4f} ({dataset_results['relative_improvement']:.2f}%)")
        else:
            # Print MEDTOK only
            logger.info(f"  MEDTOK accuracy: {dataset_results['medtok']['accuracy']:.4f}")
        
        # Print metrics by difficulty
        logger.info("  By difficulty:")
        
        for difficulty, metrics in dataset_results["medtok"]["by_difficulty"].items():
            logger.info(f"    {difficulty}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    main()
