#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text processor for medical code descriptions.

This script processes raw medical code descriptions to enhance them with
clinical context, standardize formatting, and enrich sparse descriptions.
"""

import os
import argparse
import pandas as pd
import json
import re
import string
from tqdm import tqdm
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process medical code text descriptions")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Input CSV file with medical codes")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output CSV file for processed descriptions")
    parser.add_argument("--code_col", type=str, default="code", 
                        help="Column name for medical codes")
    parser.add_argument("--desc_col", type=str, default="description", 
                        help="Column name for descriptions")
    parser.add_argument("--system_col", type=str, default="system", 
                        help="Column name for code system (ICD9, SNOMED, etc.)")
    parser.add_argument("--enrich", action="store_true", 
                        help="Enrich sparse descriptions with external resources")
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize description formatting")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of worker threads for parallel processing")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for processing")
    
    return parser.parse_args()


def clean_text(text):
    """
    Clean and standardize text descriptions.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s\.,;:\-\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize formatting
    text = text.strip()
    
    return text


def standardize_description(row, code_col, desc_col, system_col):
    """
    Standardize medical code descriptions based on code system.
    
    Args:
        row: DataFrame row
        code_col: Column name for medical codes
        desc_col: Column name for descriptions
        system_col: Column name for code system
        
    Returns:
        Standardized description
    """
    code = row[code_col]
    description = row[desc_col]
    system = row[system_col] if system_col in row and row[system_col] else "UNKNOWN"
    
    # Clean the description
    description = clean_text(description)
    
    # Handle based on code system
    if system.upper() in ["ICD9", "ICD9CM", "ICD-9", "ICD-9-CM"]:
        if not description.startswith("diagnosis:") and not description.startswith("procedure:"):
            if code.startswith("V") or code.startswith("E"):
                description = f"supplementary classification: {description}"
            elif "." in code and float(code.split(".")[0]) >= 1 and float(code.split(".")[0]) <= 999:
                description = f"diagnosis: {description}"
            else:
                description = f"procedure: {description}"
    
    elif system.upper() in ["ICD10", "ICD10CM", "ICD-10", "ICD-10-CM"]:
        if not description.startswith("diagnosis:"):
            description = f"diagnosis: {description}"
    
    elif system.upper() in ["ICD10PCS", "ICD-10-PCS"]:
        if not description.startswith("procedure:"):
            description = f"procedure: {description}"
    
    elif system.upper() in ["CPT", "HCPCS"]:
        if not description.startswith("procedure:"):
            description = f"procedure: {description}"
    
    elif system.upper() in ["LOINC"]:
        if not description.startswith("laboratory:"):
            description = f"laboratory: {description}"
    
    elif system.upper() in ["RXNORM", "NDC", "ATC"]:
        if not description.startswith("medication:"):
            description = f"medication: {description}"
    
    return description


def enrich_description(row, code_col, desc_col, system_col):
    """
    Enrich sparse descriptions with additional clinical context.
    
    Args:
        row: DataFrame row
        code_col: Column name for medical codes
        desc_col: Column name for descriptions
        system_col: Column name for code system
        
    Returns:
        Enriched description
    """
    code = row[code_col]
    description = row[desc_col]
    system = row[system_col] if system_col in row and row[system_col] else "UNKNOWN"
    
    # Skip if description is already detailed enough
    if len(description.split()) > 10:
        return description
    
    # Try to enrich based on code system
    enriched_description = description
    
    try:
        # For medications, add drug class, route, and common uses
        if system.upper() in ["RXNORM", "NDC", "ATC"]:
            if "antibiotic" in description.lower():
                enriched_description += ". An antibiotic medication used to treat bacterial infections."
            elif "antiviral" in description.lower():
                enriched_description += ". An antiviral medication used to treat viral infections."
            elif "insulin" in description.lower():
                enriched_description += ". A hormone medication used to control blood glucose levels in diabetes."
            elif "statin" in description.lower() or "vastatin" in description.lower():
                enriched_description += ". A cholesterol-lowering medication that blocks the production of cholesterol in the liver."
            elif "ace inhibitor" in description.lower() or "sartan" in description.lower():
                enriched_description += ". A medication used to lower blood pressure by relaxing blood vessels."
        
        # For diagnoses, add definition and common symptoms
        elif system.upper() in ["ICD9", "ICD10", "ICD9CM", "ICD10CM", "SNOMED"]:
            if "diabetes" in description.lower() and "type 2" in description.lower():
                enriched_description += ". A chronic metabolic disorder characterized by high blood sugar due to insulin resistance."
            elif "diabetes" in description.lower() and "type 1" in description.lower():
                enriched_description += ". An autoimmune condition where the pancreas produces little or no insulin."
            elif "hypertension" in description.lower() or "high blood pressure" in description.lower():
                enriched_description += ". A condition in which the force of blood against artery walls is too high."
            elif "pneumonia" in description.lower():
                enriched_description += ". An infection that inflames the air sacs in one or both lungs, which may fill with fluid."
            elif "myocardial infarction" in description.lower() or "heart attack" in description.lower():
                enriched_description += ". A medical emergency where blood flow to part of the heart is blocked, causing tissue damage."
    except Exception as e:
        logger.warning(f"Error enriching description for code {code}: {e}")
    
    return enriched_description


def process_batch(batch, args):
    """
    Process a batch of medical codes.
    
    Args:
        batch: DataFrame batch
        args: Command line arguments
        
    Returns:
        Processed batch
    """
    processed_batch = batch.copy()
    
    for idx, row in batch.iterrows():
        # Standardize if requested
        if args.standardize:
            processed_batch.at[idx, args.desc_col] = standardize_description(
                row, args.code_col, args.desc_col, args.system_col
            )
        
        # Enrich if requested
        if args.enrich:
            processed_batch.at[idx, args.desc_col] = enrich_description(
                processed_batch.loc[idx], args.code_col, args.desc_col, args.system_col
            )
    
    return processed_batch


def main():
    """Main processing function."""
    # Parse arguments
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    required_columns = [args.code_col, args.desc_col]
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Input file must contain columns: {', '.join(required_columns)}")
        return
    
    # Add system column if it doesn't exist
    if args.system_col not in df.columns:
        logger.warning(f"System column {args.system_col} not found. Adding with default value 'UNKNOWN'.")
        df[args.system_col] = "UNKNOWN"
    
    # Process data in batches with parallel execution
    logger.info(f"Processing {len(df)} medical codes...")
    
    # Split data into batches
    batches = [df.iloc[i:i+args.batch_size] for i in range(0, len(df), args.batch_size)]
    
    # Process batches in parallel
    processed_batches = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_batch, batch, args): i for i, batch in enumerate(batches)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_idx = futures[future]
            try:
                processed_batch = future.result()
                processed_batches.append((batch_idx, processed_batch))
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
    
    # Combine processed batches
    processed_batches.sort(key=lambda x: x[0])
    processed_df = pd.concat([batch for _, batch in processed_batches], ignore_index=True)
    
    # Save processed data
    logger.info(f"Saving processed data to {args.output}...")
    try:
        if args.output.endswith('.csv'):
            processed_df.to_csv(args.output, index=False)
        elif args.output.endswith('.json'):
            processed_json = processed_df.to_dict(orient='records')
            with open(args.output, 'w') as f:
                json.dump(processed_json, f, indent=2)
        else:
            logger.error("Unsupported output file format. Use CSV or JSON.")
            return
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        return
    
    logger.info(f"Processing completed. Processed {len(processed_df)} medical codes.")


if __name__ == "__main__":
    main()
