"""
FinBERT Sentiment Analysis Script
Classifies financial sentiment of extracted statements using ProsusAI/finbert.
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Configuration
MODEL_NAME = "ProsusAI/finbert"
INPUT_DIR = Path("Data/llm_outputs")
OUTPUT_DIR = Path("Data/sentiment_outputs")

def setup_pipeline():
    """Load FinBERT model and tokenizer"""
    print(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        return nlp
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def process_file(nlp, file_path: Path, output_path: Path):
    """Process a single JSON file and save CSV output"""
    print(f"Processing {file_path.name}...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        company = data.get("company", file_path.stem)
        
        # Categories to process
        categories = ["positive", "negative", "forward_looking", "risks"]
        
        results = []
        
        for category in categories:
            statements = data.get(category, [])
            if not statements:
                continue
                
            # Process strictly as a list of strings
            # If for some reason it's a string (though it shouldn't be per previous step), wrap it
            if isinstance(statements, str):
                statements = [statements]
                
            # Batch processing for efficiency is handled by pipeline if passed a list,
            # but for clarity and error handling we'll do it or simple list comprehension.
            # Pipeline is faster on list.
            
            try:
                # Truncation is handled by tokenizer but we should explicitely ensure max length isn't an issue
                # FinBERT max length is usually 512.
                # Pipeline handles this but might warn.
                
                # Filter out empty statements
                valid_statements = [s for s in statements if s and isinstance(s, str) and s.strip()]
                
                if not valid_statements:
                    continue

                predictions = nlp(valid_statements, padding=True, truncation=True, max_length=512)
                
                for stmt, pred in zip(valid_statements, predictions):
                    results.append({
                        "company": company,
                        "statement": stmt,
                        "category": category,
                        "sentiment": pred["label"], # Positive, Negative, Neutral
                        "confidence": pred["score"]
                    })
                    
            except Exception as e:
                print(f"  Error processing category {category} in {file_path.name}: {e}")
                
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"  Saved {len(df)} rows to {output_path.name}")
        else:
            print(f"  No valid statements found in {file_path.name}")
            
    except Exception as e:
        print(f"  Error processing content of {file_path.name}: {e}")

def main():
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} does not exist.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Model
    nlp = setup_pipeline()
    
    # 2. Get Files
    files = sorted(list(INPUT_DIR.glob("*.json")))
    if not files:
        print("No input JSON files found.")
        return
        
    print(f"Found {len(files)} files to process.")
    
    # 3. Process
    for file_path in files:
        output_file = OUTPUT_DIR / f"{file_path.stem}.csv"
        # Optional: Skip if exists? User didn't specify. We'll overwrite to be safe.
        process_file(nlp, file_path, output_file)
        
    print("\nSentiment analysis complete.")

if __name__ == "__main__":
    main()
