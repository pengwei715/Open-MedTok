#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download medical code databases.

This script downloads and processes medical code datasets from various sources,
including ICD-9, ICD-10, SNOMED CT, RxNorm, ATC, and NDC.
"""

import os
import argparse
import requests
import zipfile
import gzip
import shutil
import pandas as pd
import json
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
    parser = argparse.ArgumentParser(description="Download medical code databases")
    
    parser.add_argument("--output", type=str, required=True, 
                        help="Output directory for downloaded files")
    parser.add_argument("--icd9", action="store_true", 
                        help="Download ICD-9-CM codes")
    parser.add_argument("--icd10", action="store_true", 
                        help="Download ICD-10-CM codes")
    parser.add_argument("--snomed", action="store_true", 
                        help="Download SNOMED CT codes (requires credentials)")
    parser.add_argument("--rxnorm", action="store_true", 
                        help="Download RxNorm codes")
    parser.add_argument("--atc", action="store_true", 
                        help="Download ATC codes")
    parser.add_argument("--ndc", action="store_true", 
                        help="Download NDC codes")
    parser.add_argument("--cpt", action="store_true", 
                        help="Download sample CPT codes (not full set)")
    parser.add_argument("--all", action="store_true", 
                        help="Download all available code sets")
    parser.add_argument("--umls_username", type=str, default="", 
                        help="UMLS username (for SNOMED CT)")
    parser.add_argument("--umls_password", type=str, default="", 
                        help="UMLS password (for SNOMED CT)")
    
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


def download_icd9(output_dir):
    """
    Download ICD-9-CM codes.
    
    Args:
        output_dir: Output directory
    """
    logger.info("Downloading ICD-9-CM codes...")
    
    # Create output directory
    icd9_dir = os.path.join(output_dir, "icd9")
    os.makedirs(icd9_dir, exist_ok=True)
    
    # URLs
    urls = {
        "diagnoses": "https://raw.githubusercontent.com/kamillamagna/ICD-9-CM/master/codes.csv",
        "procedures": "https://raw.githubusercontent.com/kamillamagna/ICD-9-CM/master/procedures.csv"
    }
    
    # Download files
    for name, url in urls.items():
        output_file = os.path.join(icd9_dir, f"{name}.csv")
        download_file(url, output_file, f"Downloading ICD-9-CM {name}")
    
    # Process and combine files
    try:
        diagnoses_df = pd.read_csv(os.path.join(icd9_dir, "diagnoses.csv"))
        procedures_df = pd.read_csv(os.path.join(icd9_dir, "procedures.csv"))
        
        # Add type column
        diagnoses_df['type'] = 'diagnosis'
        procedures_df['type'] = 'procedure'
        
        # Combine
        combined_df = pd.concat([diagnoses_df, procedures_df])
        
        # Save combined file
        combined_df.to_csv(os.path.join(icd9_dir, "icd9_combined.csv"), index=False)
        
        logger.info(f"ICD-9-CM codes downloaded and processed. Total codes: {len(combined_df)}")
        
    except Exception as e:
        logger.error(f"Error processing ICD-9-CM files: {e}")


def download_icd10(output_dir):
    """
    Download ICD-10-CM and ICD-10-PCS codes.
    
    Args:
        output_dir: Output directory
    """
    logger.info("Downloading ICD-10 codes...")
    
    # Create output directory
    icd10_dir = os.path.join(output_dir, "icd10")
    os.makedirs(icd10_dir, exist_ok=True)
    
    # URLs
    urls = {
        "cm": "https://www.cms.gov/medicare/icd-10/2023-icd-10-cm/downloads/2023-icd-10-cm-codes.zip",
        "pcs": "https://www.cms.gov/medicare/icd-10/2023-icd-10-pcs/downloads/2023-icd-10-pcs-codes.zip"
    }
    
    for name, url in urls.items():
        zip_file = os.path.join(icd10_dir, f"{name}.zip")
        extract_dir = os.path.join(icd10_dir, name)
        
        # Download zip file
        download_file(url, zip_file, f"Downloading ICD-10-{name.upper()}")
        
        # Extract zip file
        os.makedirs(extract_dir, exist_ok=True)
        extract_zip(zip_file, extract_dir)
    
    # Process files
    try:
        # Process ICD-10-CM
        cm_files = [f for f in os.listdir(os.path.join(icd10_dir, "cm")) if f.endswith(".txt")]
        cm_data = []
        
        for file in cm_files:
            file_path = os.path.join(icd10_dir, "cm", file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        code, desc = parts
                        cm_data.append({
                            "code": code.strip(),
                            "description": desc.strip(),
                            "system": "ICD10CM",
                            "type": "diagnosis"
                        })
        
        # Process ICD-10-PCS
        pcs_files = [f for f in os.listdir(os.path.join(icd10_dir, "pcs")) if f.endswith(".txt")]
        pcs_data = []
        
        for file in pcs_files:
            file_path = os.path.join(icd10_dir, "pcs", file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        code, desc = parts
                        pcs_data.append({
                            "code": code.strip(),
                            "description": desc.strip(),
                            "system": "ICD10PCS",
                            "type": "procedure"
                        })
        
        # Save to CSV
        pd.DataFrame(cm_data).to_csv(os.path.join(icd10_dir, "icd10cm.csv"), index=False)
        pd.DataFrame(pcs_data).to_csv(os.path.join(icd10_dir, "icd10pcs.csv"), index=False)
        
        # Combine
        combined_df = pd.concat([pd.DataFrame(cm_data), pd.DataFrame(pcs_data)])
        combined_df.to_csv(os.path.join(icd10_dir, "icd10_combined.csv"), index=False)
        
        logger.info(f"ICD-10 codes downloaded and processed. Total codes: {len(combined_df)}")
        
    except Exception as e:
        logger.error(f"Error processing ICD-10 files: {e}")


def download_rxnorm(output_dir):
    """
    Download RxNorm codes.
    
    Args:
        output_dir: Output directory
    """
    logger.info("Downloading RxNorm codes...")
    
    # Create output directory
    rxnorm_dir = os.path.join(output_dir, "rxnorm")
    os.makedirs(rxnorm_dir, exist_ok=True)
    
    # URL for current RxNorm release
    url = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip"
    
    # Download zip file
    zip_file = os.path.join(rxnorm_dir, "rxnorm.zip")
    extract_dir = os.path.join(rxnorm_dir, "extract")
    
    download_file(url, zip_file, "Downloading RxNorm")
    
    # Extract zip file
    os.makedirs(extract_dir, exist_ok=True)
    extract_zip(zip_file, extract_dir)
    
    # Process files
    try:
        # Find RRF directory
        rrf_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if "rrf" in dirs:
                rrf_dir = os.path.join(root, "rrf")
                break
        
        if not rrf_dir:
            logger.error("RRF directory not found in RxNorm extract")
            return
        
        # Read RXNCONSO.RRF
        rxnconso_path = os.path.join(rrf_dir, "RXNCONSO.RRF")
        
        if not os.path.exists(rxnconso_path):
            logger.error(f"RXNCONSO.RRF not found in {rrf_dir}")
            return
        
        # Column names for RXNCONSO.RRF
        columns = [
            "RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "RXAUI",
            "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL",
            "SUPPRESS", "CVF"
        ]
        
        # Read file
        df = pd.read_csv(
            rxnconso_path,
            sep='|',
            header=None,
            names=columns,
            index_col=False,
            encoding='utf-8',
            errors='ignore'
        )
        
        # Filter to active preferred terms
        df = df[(df['SUPPRESS'] == 'N') & (df['ISPREF'] == 'Y')]
        
        # Create simplified dataframe
        rxnorm_df = pd.DataFrame({
            'code': df['RXCUI'],
            'description': df['STR'],
            'system': 'RxNorm',
            'type': 'medication'
        })
        
        # Remove duplicates
        rxnorm_df = rxnorm_df.drop_duplicates(subset=['code'])
        
        # Save to CSV
        rxnorm_df.to_csv(os.path.join(rxnorm_dir, "rxnorm.csv"), index=False)
        
        logger.info(f"RxNorm codes downloaded and processed. Total codes: {len(rxnorm_df)}")
        
    except Exception as e:
        logger.error(f"Error processing RxNorm files: {e}")


def download_atc(output_dir):
    """
    Download ATC codes.
    
    Args:
        output_dir: Output directory
    """
    logger.info("Downloading ATC codes...")
    
    # Create output directory
    atc_dir = os.path.join(output_dir, "atc")
    os.makedirs(atc_dir, exist_ok=True)
    
    # URLs
    url = "https://www.whocc.no/filearchive/publications/2023_atc_index_with_ddds.txt"
    
    # Download file
    output_file = os.path.join(atc_dir, "atc_raw.txt")
    download_file(url, output_file, "Downloading ATC")
    
    # Process file
    try:
        atc_data = []
        
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        description = parts[1].strip()
                        
                        if code and description:
                            atc_data.append({
                                "code": code,
                                "description": description,
                                "system": "ATC",
                                "type": "medication"
                            })
        
        # Save to CSV
        atc_df = pd.DataFrame(atc_data)
        atc_df.to_csv(os.path.join(atc_dir, "atc.csv"), index=False)
        
        logger.info(f"ATC codes downloaded and processed. Total codes: {len(atc_df)}")
        
    except Exception as e:
        logger.error(f"Error processing ATC file: {e}")


def download_ndc(output_dir):
    """
    Download NDC codes.
    
    Args:
        output_dir: Output directory
    """
    logger.info("Downloading NDC codes...")
    
    # Create output directory
    ndc_dir = os.path.join(output_dir, "ndc")
    os.makedirs(ndc_dir, exist_ok=True)
    
    # URL for NDC package file
    url = "https://www.accessdata.fda.gov/cder/ndctext.zip"
    
    # Download zip file
    zip_file = os.path.join(ndc_dir, "ndc.zip")
    extract_dir = os.path.join(ndc_dir, "extract")
    
    download_file(url, zip_file, "Downloading NDC")
    
    # Extract zip file
    os.makedirs(extract_dir, exist_ok=True)
    extract_zip(zip_file, extract_dir)
    
    # Process file
    try:
        # Find product.txt file
        product_file = os.path.join(extract_dir, "product.txt")
        
        if not os.path.exists(product_file):
            logger.error(f"product.txt not found in {extract_dir}")
            return
        
        # Read file
        columns = [
            "PRODUCTID", "PRODUCTNDC", "PRODUCTTYPENAME", "PROPRIETARYNAME",
            "PROPRIETARYNAMESUFFIX", "NONPROPRIETARYNAME", "DOSAGEFORMNAME",
            "ROUTENAME", "STARTMARKETINGDATE", "ENDMARKETINGDATE", "MARKETINGCATEGORYNAME",
            "APPLICATIONNUMBER", "LABELERNAME", "SUBSTANCENAME", "ACTIVE_NUMERATOR_STRENGTH",
            "ACTIVE_INGRED_UNIT", "PHARM_CLASSES", "DEASCHEDULE", "NDC_EXCLUDE_FLAG",
            "LISTING_RECORD_CERTIFIED_THROUGH"
        ]
        
        df = pd.read_csv(
            product_file,
            sep='\t',
            header=None,
            names=columns,
            index_col=False,
            encoding='utf-8',
            errors='ignore',
            dtype=str
        )
        
        # Filter active products
        df = df[df['NDC_EXCLUDE_FLAG'] != 'E']
        
        # Create simplified dataframe
        ndc_df = pd.DataFrame({
            'code': df['PRODUCTNDC'],
            'description': df.apply(
                lambda row: f"{row['PROPRIETARYNAME']} {row['DOSAGEFORMNAME']} ({row['NONPROPRIETARYNAME']})",
                axis=1
            ),
            'system': 'NDC',
            'type': 'medication'
        })
        
        # Remove duplicates
        ndc_df = ndc_df.drop_duplicates(subset=['code'])
        
        # Save to CSV
        ndc_df.to_csv(os.path.join(ndc_dir, "ndc.csv"), index=False)
        
        logger.info(f"NDC codes downloaded and processed. Total codes: {len(ndc_df)}")
        
    except Exception as e:
        logger.error(f"Error processing NDC file: {e}")


def download_sample_cpt(output_dir):
    """
    Download sample CPT codes (not the full set which requires a license).
    
    Args:
        output_dir: Output directory
    """
    logger.info("Downloading sample CPT codes...")
    
    # Create output directory
    cpt_dir = os.path.join(output_dir, "cpt")
    os.makedirs(cpt_dir, exist_ok=True)
    
    # Create a sample of common CPT codes
    cpt_data = [
        {"code": "99201", "description": "Office/outpatient visit new", "system": "CPT", "type": "procedure"},
        {"code": "99202", "description": "Office/outpatient visit new", "system": "CPT", "type": "procedure"},
        {"code": "99203", "description": "Office/outpatient visit new", "system": "CPT", "type": "procedure"},
        {"code": "99204", "description": "Office/outpatient visit new", "system": "CPT", "type": "procedure"},
        {"code": "99205", "description": "Office/outpatient visit new", "system": "CPT", "type": "procedure"},
        {"code": "99211", "description": "Office/outpatient visit est", "system": "CPT", "type": "procedure"},
        {"code": "99212", "description": "Office/outpatient visit est", "system": "CPT", "type": "procedure"},
        {"code": "99213", "description": "Office/outpatient visit est", "system": "CPT", "type": "procedure"},
        {"code": "99214", "description": "Office/outpatient visit est", "system": "CPT", "type": "procedure"},
        {"code": "99215", "description": "Office/outpatient visit est", "system": "CPT", "type": "procedure"},
        {"code": "10021", "description": "Fna w/o image", "system": "CPT", "type": "procedure"},
        {"code": "10022", "description": "Fna w/image", "system": "CPT", "type": "procedure"},
        {"code": "10040", "description": "Acne surgery", "system": "CPT", "type": "procedure"},
        {"code": "10060", "description": "Drainage of skin abscess", "system": "CPT", "type": "procedure"},
        {"code": "10061", "description": "Drainage of skin abscess", "system": "CPT", "type": "procedure"},
        {"code": "10080", "description": "Drainage of pilonidal cyst", "system": "CPT", "type": "procedure"},
        {"code": "10081", "description": "Drainage of pilonidal cyst", "system": "CPT", "type": "procedure"},
        {"code": "10120", "description": "Remove foreign body", "system": "CPT", "type": "procedure"},
        {"code": "10121", "description": "Remove foreign body", "system": "CPT", "type": "procedure"},
        {"code": "10140", "description": "Drainage of hematoma/fluid", "system": "CPT", "type": "procedure"},
        {"code": "10160", "description": "Puncture drainage of lesion", "system": "CPT", "type": "procedure"},
        {"code": "10180", "description": "Complex drainage, wound", "system": "CPT", "type": "procedure"},
        {"code": "11000", "description": "Debride infected skin", "system": "CPT", "type": "procedure"},
        {"code": "11001", "description": "Debride infected skin add-on", "system": "CPT", "type": "procedure"},
        {"code": "11010", "description": "Debride skin, fx", "system": "CPT", "type": "procedure"},
        {"code": "11011", "description": "Debride skin/muscle, fx", "system": "CPT", "type": "procedure"},
        {"code": "11012", "description": "Debride skin/muscle/bone, fx", "system": "CPT", "type": "procedure"},
        {"code": "11042", "description": "Deb subq tissue 20 sq cm/<", "system": "CPT", "type": "procedure"},
        {"code": "11043", "description": "Deb musc/fascia 20 sq cm/<", "system": "CPT", "type": "procedure"},
        {"code": "11044", "description": "Deb bone 20 sq cm/<", "system": "CPT", "type": "procedure"},
        {"code": "11045", "description": "Deb subq tissue add-on", "system": "CPT", "type": "procedure"},
        {"code": "11046", "description": "Deb musc/fascia add-on", "system": "CPT", "type": "procedure"},
        {"code": "11047", "description": "Deb bone add-on", "system": "CPT", "type": "procedure"},
        {"code": "11055", "description": "Trim skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11056", "description": "Trim skin lesions, 2 to 4", "system": "CPT", "type": "procedure"},
        {"code": "11057", "description": "Trim skin lesions, over 4", "system": "CPT", "type": "procedure"},
        {"code": "11100", "description": "Biopsy, skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11101", "description": "Biopsy, skin add-on", "system": "CPT", "type": "procedure"},
        {"code": "11200", "description": "Removal of skin tags", "system": "CPT", "type": "procedure"},
        {"code": "11201", "description": "Remove skin tags add-on", "system": "CPT", "type": "procedure"},
        {"code": "11300", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11301", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11302", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11303", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11305", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11306", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11307", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11308", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11310", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11311", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11312", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11313", "description": "Shave skin lesion", "system": "CPT", "type": "procedure"},
        {"code": "11400", "description": "Exc tr-ext b9+marg 0.5 < cm", "system": "CPT", "type": "procedure"},
        {"code": "11401", "description": "Exc tr-ext b9+marg 0.6-1 cm", "system": "CPT", "type": "procedure"},
        {"code": "11402", "description": "Exc tr-ext b9+marg 1.1-2 cm", "system": "CPT", "type": "procedure"},
        {"code": "11403", "description": "Exc tr-ext b9+marg 2.1-3 cm", "system": "CPT", "type": "procedure"},
        {"code": "11404", "description": "Exc tr-ext b9+marg 3.1-4 cm", "system": "CPT", "type": "procedure"},
        {"code": "11406", "description": "Exc tr-ext b9+marg > 4.0 cm", "system": "CPT", "type": "procedure"}
    ]
    
    # Save to CSV
    cpt_df = pd.DataFrame(cpt_data)
    cpt_df.to_csv(os.path.join(cpt_dir, "cpt_sample.csv"), index=False)
    
    logger.info(f"Sample CPT codes created. Total codes: {len(cpt_df)}")
    
    # Note about full CPT codes
    logger.info("Note: This is only a small sample of CPT codes. The full set requires a license from the AMA.")


def download_snomed(output_dir, username, password):
    """
    Download SNOMED CT codes.
    
    Args:
        output_dir: Output directory
        username: UMLS username
        password: UMLS password
    """
    if not username or not password:
        logger.error("UMLS username and password required for SNOMED CT download")
        return
    
    logger.info("Downloading SNOMED CT codes...")
    
    # Create output directory
    snomed_dir = os.path.join(output_dir, "snomed")
    os.makedirs(snomed_dir, exist_ok=True)
    
    # Login to UMLS
    try:
        # Create a session
        session = requests.Session()
        
        # Get the login page
        login_url = "https://utslogin.nlm.nih.gov/cas/login"
        response = session.get(login_url)
        
        # Extract the execution value
        import re
        execution = re.search(r'<input type="hidden" name="execution" value="([^"]+)"', response.text).group(1)
        
        # Login
        login_data = {
            'username': username,
            'password': password,
            'execution': execution,
            '_eventId': 'submit'
        }
        
        response = session.post(login_url, data=login_data)
        
        if "TGT created" not in response.text:
            logger.error("UMLS login failed. Check your credentials.")
            return
        
        # Get service ticket
        service_url = "https://download.nlm.nih.gov/umls/kss/IHTSDO20220731/SnomedCT_InternationalRF2_PRODUCTION_20220731T120000Z.zip"
        tgt_url = response.url
        
        service_ticket_url = f"{tgt_url}?service={service_url}"
        response = session.get(service_ticket_url)
        service_ticket = response.text
        
        # Download SNOMED CT
        download_url = f"{service_url}?ticket={service_ticket}"
        zip_file = os.path.join(snomed_dir, "snomed.zip")
        
        logger.info("Downloading SNOMED CT (this may take a while)...")
        urllib.request.urlretrieve(download_url, zip_file)
        
        # Extract zip file
        extract_dir = os.path.join(snomed_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info("Extracting SNOMED CT...")
        extract_zip(zip_file, extract_dir)
        
        # Process files
        try:
            # Find Snapshot directory
            snapshot_dir = None
            for root, dirs, files in os.walk(extract_dir):
                if "Snapshot" in dirs:
                    snapshot_dir = os.path.join(root, "Snapshot")
                    break
            
            if not snapshot_dir:
                logger.error("Snapshot directory not found in SNOMED CT extract")
                return
            
            # Find Terminology directory
            terminology_dir = None
            for root, dirs, files in os.walk(snapshot_dir):
                if "Terminology" in dirs:
                    terminology_dir = os.path.join(root, "Terminology")
                    break
            
            if not terminology_dir:
                logger.error("Terminology directory not found in SNOMED CT extract")
                return
            
            # Read Description file
            description_file = os.path.join(terminology_dir, "sct2_Description_Snapshot-en_INT_20220731.txt")
            
            if not os.path.exists(description_file):
                logger.error(f"Description file not found in {terminology_dir}")
                return
            
            # Read file
            columns = [
                "id", "effectiveTime", "active", "moduleId", "conceptId",
                "languageCode", "typeId", "term", "caseSignificanceId"
            ]
            
            df = pd.read_csv(
                description_file,
                sep='\t',
                header=0,
                names=columns,
                index_col=False,
                encoding='utf-8',
                errors='ignore'
            )
            
            # Filter active preferred terms
            df = df[(df['active'] == 1) & (df['typeId'] == 900000000000003001)]
            
            # Create simplified dataframe
            snomed_df = pd.DataFrame({
                'code': df['conceptId'],
                'description': df['term'],
                'system': 'SNOMED',
                'type': 'diagnosis'  # Simplified, could be more specific
            })
            
            # Remove duplicates
            snomed_df = snomed_df.drop_duplicates(subset=['code'])
            
            # Save to CSV
            snomed_df.to_csv(os.path.join(snomed_dir, "snomed.csv"), index=False)
            
            logger.info(f"SNOMED CT codes downloaded and processed. Total codes: {len(snomed_df)}")
            
        except Exception as e:
            logger.error(f"Error processing SNOMED CT files: {e}")
    
    except Exception as e:
        logger.error(f"Error downloading SNOMED CT: {e}")


def combine_all_codes(output_dir):
    """
    Combine all downloaded code systems into a single file.
    
    Args:
        output_dir: Output directory
    """
    logger.info("Combining all code systems...")
    
    # Create list to store all dataframes
    all_dfs = []
    
    # ICD-9
    icd9_path = os.path.join(output_dir, "icd9", "icd9_combined.csv")
    if os.path.exists(icd9_path):
        try:
            icd9_df = pd.read_csv(icd9_path)
            icd9_df['system'] = 'ICD9'
            all_dfs.append(icd9_df)
        except Exception as e:
            logger.error(f"Error reading ICD-9 data: {e}")
    
    # ICD-10
    icd10_path = os.path.join(output_dir, "icd10", "icd10_combined.csv")
    if os.path.exists(icd10_path):
        try:
            icd10_df = pd.read_csv(icd10_path)
            all_dfs.append(icd10_df)
        except Exception as e:
            logger.error(f"Error reading ICD-10 data: {e}")
    
    # RxNorm
    rxnorm_path = os.path.join(output_dir, "rxnorm", "rxnorm.csv")
    if os.path.exists(rxnorm_path):
        try:
            rxnorm_df = pd.read_csv(rxnorm_path)
            all_dfs.append(rxnorm_df)
        except Exception as e:
            logger.error(f"Error reading RxNorm data: {e}")
    
    # ATC
    atc_path = os.path.join(output_dir, "atc", "atc.csv")
    if os.path.exists(atc_path):
        try:
            atc_df = pd.read_csv(atc_path)
            all_dfs.append(atc_df)
        except Exception as e:
            logger.error(f"Error reading ATC data: {e}")
    
    # NDC
    ndc_path = os.path.join(output_dir, "ndc", "ndc.csv")
    if os.path.exists(ndc_path):
        try:
            ndc_df = pd.read_csv(ndc_path)
            all_dfs.append(ndc_df)
        except Exception as e:
            logger.error(f"Error reading NDC data: {e}")
    
    # CPT
    cpt_path = os.path.join(output_dir, "cpt", "cpt_sample.csv")
    if os.path.exists(cpt_path):
        try:
            cpt_df = pd.read_csv(cpt_path)
            all_dfs.append(cpt_df)
        except Exception as e:
            logger.error(f"Error reading CPT data: {e}")
    
    # SNOMED CT
    snomed_path = os.path.join(output_dir, "snomed", "snomed.csv")
    if os.path.exists(snomed_path):
        try:
            snomed_df = pd.read_csv(snomed_path)
            all_dfs.append(snomed_df)
        except Exception as e:
            logger.error(f"Error reading SNOMED CT data: {e}")
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined data
        combined_df.to_csv(os.path.join(output_dir, "all_medical_codes.csv"), index=False)
        
        # Save as JSON as well
        combined_df.to_json(os.path.join(output_dir, "all_medical_codes.json"), orient="records", indent=2)
        
        logger.info(f"All code systems combined. Total codes: {len(combined_df)}")
    else:
        logger.error("No code systems found to combine")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Download code systems
    if args.all or args.icd9:
        download_icd9(args.output)
    
    if args.all or args.icd10:
        download_icd10(args.output)
    
    if args.all or args.rxnorm:
        download_rxnorm(args.output)
    
    if args.all or args.atc:
        download_atc(args.output)
    
    if args.all or args.ndc:
        download_ndc(args.output)
    
    if args.all or args.cpt:
        download_sample_cpt(args.output)
    
    if args.all or args.snomed:
        download_snomed(args.output, args.umls_username, args.umls_password)
    
    # Combine all code systems
    combine_all_codes(args.output)
    
    logger.info("Download complete")


if __name__ == "__main__":
    main()
