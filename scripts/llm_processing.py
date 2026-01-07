"""
LLM Processing Script
Extracts financial insights from annual reports using OpenAI API.
"""

import os
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' library not found. Please install it using 'pip install openai'")
    sys.exit(1)

# Configuration
API_KEY = "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"
# Configuration
API_KEY = "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"
# Optimized model list - removing invalid/too-small models
MODELS_TO_TRY = ["gpt-4o-mini", "gpt-4o"]
INPUT_DIR = Path("Data/cleaned_text")
OUTPUT_DIR = Path("Data/llm_outputs")

def get_openai_client():
    return OpenAI(api_key=API_KEY)

def construct_prompt(report_text: str) -> str:
    return f"""You are a financial analyst.

From the following annual report text, extract:

1. Key positive statements
2. Key negative statements
3. Forward-looking guidance
4. Risk-related commentary

Rules:
- Return STRICTLY valid JSON
- No explanations
- No markdown formatting (no ```json ... ```)
- Each item must be a short, standalone statement

JSON format:
{{
  "positive": [],
  "negative": [],
  "forward_looking": [],
  "risks": []
}}
    
Annual Report Text:
{report_text}
"""

def merge_results(results: list[dict]) -> dict:
    """Merge multiple JSON results into one."""
    final = {
        "positive": [],
        "negative": [],
        "forward_looking": [],
        "risks": []
    }
    for res in results:
        for key in final:
            if key in res and isinstance(res[key], list):
                final[key].extend(res[key])
    return final

def process_chunk(client: OpenAI, model: str, text: str, chunk_index: int, total_chunks: int) -> Optional[Dict[str, Any]]:
    print(f"    Processing chunk {chunk_index + 1}/{total_chunks} ({len(text)} chars)...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": construct_prompt(text)}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        return data

    except Exception as e:
        print(f"    Error processing chunk {chunk_index + 1}: {e}")
        return None

def process_file_with_model(client: OpenAI, file_path: Path, model: str, text: str) -> Optional[Dict[str, Any]]:
    # This function is replaced by process_file_smart logic but kept for safety if called elsewhere, 
    # though process_file_smart handles the logic now.
    pass

# Adjusted chunk size for safety
CHUNK_SIZE_CHARS = 300000 

def process_file_smart(client: OpenAI, file_path: Path) -> Optional[Dict[str, Any]]:
    print(f"Processing {file_path.name}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None

    text_len = len(text)
    print(f"  Text length: {text_len} characters")
    
    # Determine chunks
    chunks = []
    if text_len > CHUNK_SIZE_CHARS:
        print(f"  File output exceeds context limit. Splitting into {text_len // CHUNK_SIZE_CHARS + 1} chunks.")
        for i in range(0, text_len, CHUNK_SIZE_CHARS):
            chunks.append(text[i : i + CHUNK_SIZE_CHARS])
    else:
        chunks.append(text)

    # Try models in order
    for model in MODELS_TO_TRY:
        chunk_results = []
        model_failed = False
        
        print(f"  Attempting with model: {model}...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Add a small delay for large operations if needed, but not strictly required unless hitting rate limits
                res = process_chunk(client, model, chunk, i, len(chunks))
                if res:
                    chunk_results.append(res)
                else:
                    model_failed = True
                    break
            except Exception as e:
                # API error (like model not found)
                if "model" in str(e).lower() or "unsupported" in str(e).lower():
                    print(f"  Model {model} unavailable.")
                    model_failed = True
                    break
                print(f"  Error with {model}: {e}")
                model_failed = True
                break
        
        if not model_failed and len(chunk_results) == len(chunks):
            # Success
            merged = merge_results(chunk_results)
            return {"company": file_path.stem, **merged}
        
        if "model" in str(model_failed).lower():
             continue # Try next model
             
    print("  All models failed.")
    return None

def main():
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} does not exist.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    client = get_openai_client()
    
    files = sorted(list(INPUT_DIR.glob("*.txt")))
    if not files:
        print("No input files found.")
        return
        
    print(f"Found {len(files)} files to process.")
    
    for file_path in files:
        output_file = OUTPUT_DIR / f"{file_path.stem}.json"
        
        # Skip if already exists
        if output_file.exists():
            print(f"Skipping {file_path.name} (Output exists)")
            continue
            
        result = process_file_smart(client, file_path)
        
        if result:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  Success: Saved to {output_file.name}")
            
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
