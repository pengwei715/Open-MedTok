#!/usr/bin/env python
"""
Unified preprocessing pipeline for MedTok.

This script provides an end-to-end pipeline for preparing data for MedTok:
1. Download and process medical codes
2. Download PrimeKG knowledge graph
3. Process text descriptions
4. Generate subgraphs for medical codes
5. Prepare training dataset

Usage:
    python preprocess_pipeline.py --output data/ --codes_only  # Only process codes
    python preprocess_pipeline.py --output data/ --full  # Full preprocessing
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_command(cmd, desc=None, check=True):
    """Run a command and log its output"""
    if desc:
        logger.info(f"Step: {desc}")
    
    logger.info(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        for line in result.stdout.splitlines():
            logger.info(line)
        
        for line in result.stderr.splitlines():
            if check:
                logger.warning(line)
            else:
                logger.error(line)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")
        
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)
        
        if check:
            raise
        
        return False

def verify_paths(paths):
    """Verify that all paths in the list exist"""
    for path in paths:
        if not os.path.exists(path):
            logger.error(f"Path not found: {path}")
            return False
    return True

def setup_directories(base_dir):
    """Create necessary directories for preprocessing"""
    dirs = {
        "medical_codes": os.path.join(base_dir, "medical_codes"),
        "primekg": os.path.join(base_dir, "primekg"),
        "processed_codes": os.path.join(base_dir, "processed_codes"),
        "text_descriptions": os.path.join(base_dir, "text_descriptions"),
        "graphs": os.path.join(base_dir, "graphs"),
        "dataset": os.path.join(base_dir, "dataset"),
        "logs": os.path.join(base_dir, "logs")
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return dirs

def download_medical_codes(output_dir):
    """Download medical codes"""
    return run_command(
        ["python", "scripts/download_medical_codes.py", "--output", output_dir],
        "Downloading medical codes"
    )

def download_primekg(output_dir):
    """Download PrimeKG knowledge graph"""
    return run_command(
        ["python", "scripts/download_primekg.py", "--output", output_dir],
        "Downloading PrimeKG knowledge graph"
    )

def process_medical_codes(input_dir, output_dir):
    """Process raw medical codes"""
    return run_command(
        ["python", "scripts/process_medical_codes.py", 
         "--input", input_dir, 
         "--output", output_dir],
        "Processing medical codes"
    )

def process_text_descriptions(codes_path, output_path):
    """Process text descriptions for medical codes"""
    return run_command(
        ["python", "data/preprocessing/text_processor.py",
         "--input", codes_path,
         "--output", output_path],
        "Processing text descriptions"
    )

def process_graphs(codes_path, kg_path, output_dir):
    """Process knowledge graphs for medical codes"""
    return run_command(
        ["python", "data/preprocessing/graph_processor.py",
         "--input", codes_path,
         "--kg_path", kg_path,
         "--output_dir", output_dir],
        "Processing knowledge graphs"
    )

def prepare_dataset(codes_path, text_path, graphs_dir, output_dir):
    """Prepare final dataset for training"""
    return run_command(
        ["python", "data/preprocessing/prepare_dataset.py",
         "--codes", codes_path,
         "--descriptions", text_path,
         "--graphs_dir", graphs_dir,
         "--output_dir", output_dir],
        "Preparing final dataset"
    )

def verify_dataset(dataset_dir):
    """Verify the prepared dataset"""
    train_path = os.path.join(dataset_dir, "train.json")
    val_path = os.path.join(dataset_dir, "val.json")
    test_path = os.path.join(dataset_dir, "test.json")
    
    # Check if files exist
    if not verify_paths([train_path, val_path, test_path]):
        return False
    
    # Check if files have content
    try:
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        
        return len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0
    
    except Exception as e:
        logger.error(f"Error verifying dataset: {str(e)}")
        return False

def save_config(dirs, args, output_dir):
    """Save preprocessing configuration"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "arguments": vars(args),
        "directories": dirs,
        "status": "completed"
    }
    
    config_path = os.path.join(output_dir, "preprocessing_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved preprocessing configuration to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified preprocessing pipeline for MedTok")
    parser.add_argument("--output", type=str, default="data",
                        help="Base output directory")
    parser.add_argument("--codes_only", action="store_true",
                        help="Only download and process medical codes")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading data (use existing files)")
    parser.add_argument("--full", action="store_true",
                        help="Run full preprocessing pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if output files exist")
    parser.add_argument("--skip_verification", action="store_true",
                        help="Skip verification steps")
    
    args = parser.parse_args()
    
    # Setup directories
    logger.info(f"Setting up directories in {args.output}")
    dirs = setup_directories(args.output)
    
    # Track overall success
    success = True
    
    # Download and process medical codes
    if not args.skip_download or not os.path.exists(os.path.join(dirs["medical_codes"], "codes.csv")):
        success &= download_medical_codes(dirs["medical_codes"])
    else:
        logger.info("Skipping medical code download (already exists)")
    
    if success:
        success &= process_medical_codes(
            dirs["medical_codes"], 
            dirs["processed_codes"]
        )
    
    # For codes_only mode, stop here
    if args.codes_only:
        if success:
            logger.info("Medical code processing completed successfully")
            save_config(dirs, args, args.output)
        else:
            logger.error("Medical code processing failed")
        return success
    
    # For full processing, continue with knowledge graph and text descriptions
    if args.full:
        # Download PrimeKG
        if not args.skip_download or not os.path.exists(os.path.join(dirs["primekg"], "kg.json")):
            success &= download_primekg(dirs["primekg"])
        else:
            logger.info("Skipping PrimeKG download (already exists)")
        
        # Process text descriptions
        processed_codes_path = os.path.join(dirs["processed_codes"], "codes.csv")
        text_desc_path = os.path.join(dirs["text_descriptions"], "descriptions.csv")
        
        if success and (args.force or not os.path.exists(text_desc_path)):
            success &= process_text_descriptions(processed_codes_path, text_desc_path)
        else:
            logger.info("Skipping text description processing (already exists)")
        
        # Process graphs
        kg_path = os.path.join(dirs["primekg"], "kg.json")
        
        if success and (args.force or not os.path.exists(dirs["graphs"]) or len(os.listdir(dirs["graphs"])) == 0):
            success &= process_graphs(processed_codes_path, kg_path, dirs["graphs"])
        else:
            logger.info("Skipping graph processing (already exists)")
        
        # Prepare final dataset
        if success and (args.force or not os.path.exists(os.path.join(dirs["dataset"], "train.json"))):
            success &= prepare_dataset(
                processed_codes_path,
                text_desc_path,
                dirs["graphs"],
                dirs["dataset"]
            )
        else:
            logger.info("Skipping dataset preparation (already exists)")
        
        # Verify dataset
        if success and not args.skip_verification:
            success &= verify_dataset(dirs["dataset"])
    
    # Save configuration
    save_config(dirs, args, args.output)
    
    if success:
        logger.info("Preprocessing completed successfully")
    else:
        logger.error("Preprocessing failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)