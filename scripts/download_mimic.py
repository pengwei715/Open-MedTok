#!/usr/bin/env python
"""
Script to download and prepare MIMIC-III and MIMIC-IV datasets for MedTok.
Note: This script requires PhysioNet credentials as MIMIC datasets require approval.
"""

import os
import argparse
import subprocess
import getpass
import sys
import json
from pathlib import Path

def check_dependencies():
    """Check if wget is installed"""
    try:
        subprocess.run(["wget", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def download_mimic(mimic_version, output_dir, username=None, password=None):
    """
    Download MIMIC datasets using PhysioNet credentials.
    
    Args:
        mimic_version: Either "mimic3" or "mimic4"
        output_dir: Directory to save the dataset
        username: PhysioNet username
        password: PhysioNet password
    """
    if mimic_version == "mimic3":
        dataset_url = "https://physionet.org/files/mimiciii/1.4/"
        dataset_name = "MIMIC-III"
    elif mimic_version == "mimic4":
        dataset_url = "https://physionet.org/files/mimiciv/2.2/"
        dataset_name = "MIMIC-IV"
    else:
        print(f"Unknown MIMIC version: {mimic_version}")
        return False
    
    print(f"Downloading {dataset_name} dataset")
    print("NOTE: This requires PhysioNet credentials and approved access to the dataset")
    
    # Get credentials if not provided
    if not username:
        username = input("PhysioNet username: ")
    if not password:
        password = getpass.getpass("PhysioNet password: ")
    
    # Create output directory
    output_path = os.path.join(output_dir, mimic_version)
    os.makedirs(output_path, exist_ok=True)
    
    # Download command using wget
    cmd = [
        "wget", "-r", "-N", "-c", "-np", "--user", username, "--password", password,
        "-P", output_path, dataset_url
    ]
    
    try:
        print(f"Starting download of {dataset_name}...")
        print(f"This may take several hours. Files will be saved to {output_path}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True)
        
        # Stream the output
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n{dataset_name} download completed successfully!")
            
            # Save a metadata file with dataset information
            metadata = {
                "dataset": dataset_name,
                "version": mimic_version,
                "source_url": dataset_url,
                "download_date": str(Path().absolute())
            }
            
            with open(os.path.join(output_path, "dataset_info.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return True
        else:
            print(f"\nError downloading {dataset_name}. wget returned code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error during download: {str(e)}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Download MIMIC datasets for MedTok")
    parser.add_argument("--version", type=str, choices=["mimic3", "mimic4", "all"], 
                        default="all", help="Which MIMIC version to download")
    parser.add_argument("--output", type=str, default="data", 
                        help="Directory to save datasets")
    parser.add_argument("--username", type=str, help="PhysioNet username")
    parser.add_argument("--prepare", action="store_true", 
                        help="Prepare datasets for MedTok after download")
    
    return parser.parse_args()

def prepare_dataset(mimic_version, data_dir):
    """Process downloaded MIMIC data for use with MedTok"""
    mimic_dir = os.path.join(data_dir, mimic_version)
    
    if not os.path.exists(mimic_dir):
        print(f"Error: {mimic_dir} does not exist. Please download the dataset first.")
        return False
    
    print(f"Preparing {mimic_version} dataset for use with MedTok...")
    
    # Define paths to important tables based on MIMIC version
    if mimic_version == "mimic3":
        diagnoses_table = os.path.join(mimic_dir, "physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz")
        procedures_table = os.path.join(mimic_dir, "physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv.gz")
        prescriptions_table = os.path.join(mimic_dir, "physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv.gz")
    elif mimic_version == "mimic4":
        diagnoses_table = os.path.join(mimic_dir, "physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz")
        procedures_table = os.path.join(mimic_dir, "physionet.org/files/mimiciv/2.2/hosp/procedures_icd.csv.gz")
        prescriptions_table = os.path.join(mimic_dir, "physionet.org/files/mimiciv/2.2/hosp/prescriptions.csv.gz")
    
    # Check if files exist
    for file_path in [diagnoses_table, procedures_table, prescriptions_table]:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            print("Dataset may not have been downloaded correctly.")
            return False
    
    # Create processed directory
    processed_dir = os.path.join(data_dir, f"{mimic_version}_processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process diagnoses
    print("Processing diagnoses...")
    subprocess.run([
        "python", "scripts/process_medical_codes.py",
        "--input", diagnoses_table,
        "--output", os.path.join(processed_dir, "diagnoses.csv"),
        "--format", "icd9" if mimic_version == "mimic3" else "icd10",
        "--type", "diagnosis"
    ])
    
    # Process procedures
    print("Processing procedures...")
    subprocess.run([
        "python", "scripts/process_medical_codes.py",
        "--input", procedures_table,
        "--output", os.path.join(processed_dir, "procedures.csv"),
        "--format", "icd9" if mimic_version == "mimic3" else "icd10",
        "--type", "procedure"
    ])
    
    # Process medications
    print("Processing medications...")
    subprocess.run([
        "python", "scripts/process_medical_codes.py",
        "--input", prescriptions_table,
        "--output", os.path.join(processed_dir, "medications.csv"),
        "--format", "ndc",
        "--type", "medication"
    ])
    
    print(f"Dataset preparation complete. Processed files are in {processed_dir}")
    return True

def main():
    args = parse_args()
    
    # Check for wget
    if not check_dependencies():
        print("Error: wget is required but not found. Please install wget first.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine which datasets to download
    versions = ["mimic3", "mimic4"] if args.version == "all" else [args.version]
    
    # Download each dataset
    for version in versions:
        success = download_mimic(version, args.output, args.username)
        
        if success and args.prepare:
            prepare_dataset(version, args.output)
        
        print()  # Add a blank line between datasets

if __name__ == "__main__":
    main()