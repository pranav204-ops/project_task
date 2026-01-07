"""
Main Pipeline Script
Orchestrates the entire financial report analysis pipeline:
1. PDF Extraction
2. Text Cleaning
3. LLM Insight Extraction
4. Sentiment Analysis
5. Result Aggregation
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the Scripts directory to python path to allow imports
sys.path.insert(0, str(Path(__file__).parent / "Scripts"))

try:
    from Scripts.pdf_extraction import extract_all_pdfs
    from Scripts.text_cleaning import process_text_file
    from Scripts.llm_processing import process_file_smart, get_openai_client
    from Scripts.finbert_sentiment import setup_pipeline, process_file as process_sentiment
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Fallback if Scripts is not a package or in path correctly
    try:
        from pdf_extraction import extract_all_pdfs
        from text_cleaning import process_text_file
        from llm_processing import process_file_smart, get_openai_client
        from finbert_sentiment import setup_pipeline, process_file as process_sentiment
    except ImportError as e2:
        print(f"Critical error: Could not import script modules: {e2}")
        sys.exit(1)

# Configuration - Centralized Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"

RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
CLEANED_TEXT_DIR = DATA_DIR / "cleaned_text"
LLM_OUTPUT_DIR = DATA_DIR / "llm_outputs"
SENTIMENT_OUTPUT_DIR = DATA_DIR / "sentiment_outputs"

FINAL_OUTPUT_FILE = DATA_DIR / "unified_financial_insights.csv"

def run_pdf_extraction():
    """Step 1: Extract text from PDFs"""
    print("\n--- Step 1: PDF Extraction ---")
    if not RAW_PDF_DIR.exists():
        print(f"Warning: Raw PDF directory {RAW_PDF_DIR} not found. Skipping extraction.")
        return
        
    extract_all_pdfs(
        input_dir=str(RAW_PDF_DIR),
        output_dir=str(EXTRACTED_TEXT_DIR)
    )

def run_text_cleaning():
    """Step 2: Clean extracted text"""
    print("\n--- Step 2: Text Cleaning ---")
    if not EXTRACTED_TEXT_DIR.exists():
        print("No extracted text directory found.")
        return

    CLEANED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(EXTRACTED_TEXT_DIR.glob("*.txt")))
    if not files:
        print("No extracted text files found.")
        return

    print(f"Found {len(files)} files to clean.")
    for file_path in files:
        process_text_file(file_path, CLEANED_TEXT_DIR)

def run_llm_processing():
    """Step 3: Extract insights using LLM"""
    print("\n--- Step 3: LLM Insight Extraction ---")
    if not CLEANED_TEXT_DIR.exists():
        print("No cleaned text directory found.")
        return

    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        client = get_openai_client()
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    files = sorted(list(CLEANED_TEXT_DIR.glob("*.txt")))
    if not files:
        print("No cleaned text files found.")
        return
        
    print(f"Found {len(files)} files to process with LLM.")
    
    import json
    
    for file_path in files:
        output_file = LLM_OUTPUT_DIR / f"{file_path.stem}.json"
        
        # Check if output exists to save cost/time (optional, can be removed for full reproducibility)
        if output_file.exists():
            print(f"Skipping {file_path.name} (Output exists)")
            continue
            
        result = process_file_smart(client, file_path)
        
        if result:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {output_file.name}")

def run_sentiment_analysis():
    """Step 4: Analyze sentiment with FinBERT"""
    print("\n--- Step 4: FinBERT Sentiment Analysis ---")
    if not LLM_OUTPUT_DIR.exists():
        print("No LLM output directory found.")
        return

    SENTIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    nlp = setup_pipeline()
    
    files = sorted(list(LLM_OUTPUT_DIR.glob("*.json")))
    if not files:
        print("No LLM output files found.")
        return
        
    print(f"Found {len(files)} files to analyze.")
    for file_path in files:
        output_file = SENTIMENT_OUTPUT_DIR / f"{file_path.stem}.csv"
        # We overwrite here to ensure latest model results
        process_sentiment(nlp, file_path, output_file)

def aggregate_results():
    """Step 5: Combine all results"""
    print("\n--- Step 5: Aggregating Results ---")
    if not SENTIMENT_OUTPUT_DIR.exists():
        print("No sentiment output directory found.")
        return

    files = sorted(list(SENTIMENT_OUTPUT_DIR.glob("*.csv")))
    if not files:
        print("No sentiment CSV files found.")
        return
        
    all_dfs = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Ensure column order and naming if needed
        # Expected: company, statement, category, sentiment, confidence
        # Our CSVs have this structure.
        
        combined_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"Successfully created unified dataset with {len(combined_df)} rows.")
        print(f"Saved to: {FINAL_OUTPUT_FILE}")
    else:
        print("No data to aggregate.")

def main():
    print("Starting Financial Report Analysis Pipeline...")
    
    run_pdf_extraction()
    run_text_cleaning()
    run_llm_processing()
    run_sentiment_analysis()
    aggregate_results()
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
