"""
Text Cleaning Script
Cleans extracted text for LLM readiness.
Preserves original structure, removes artifacts, and keeps the whole document intact.
"""

import re
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path to import config if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())

def remove_page_markers(text: str) -> str:
    """Remove page markers like '--- Page X ---'"""
    text = re.sub(r'---\s*Page\s+\d+\s*---', '', text, flags=re.IGNORECASE)
    return text

def remove_headers_footers(text: str, company_name: str) -> str:
    """
    Remove common headers and footers using regex patterns.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Common header/footer patterns
    # We use aggressive strictly anchored patterns to avoid deleting content
    header_patterns = [
        r'^.*\|\s*\d{4}\s+Annual\s+Report.*$',     # "Company | 2023 Annual Report"
        r'^.*Annual\s+Report\s+\d{4}.*$',          # "Annual Report 2023"
        r'^' + re.escape(company_name) + r'\s*\|\s*.*Annual Report.*$', # Specific company header
        r'^.*\|\s*Page\s*\d+.*$',                   # "Section | Page 12"
    ]
    
    # Page number patterns (standalone numbers, or "Page X", "YF 2022", etc on a single line)
    page_num_patterns = [
        r'^\d{1,4}\s*$',                            # Standalone numbers
        r'^\d{1,4}\s*YF\s*$',                       # "2022 YF"
        r'^YF\s*\d{4}\s*$',                         # "YF 2022"
        r'^Page\s+\d+\s*$',                         # "Page 12"
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('') # Preserve structure with empty lines
            continue
        
        is_artifact = False
        
        # Check explicit headers
        for pattern in header_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_artifact = True
                break
        
        # Check page numbers
        if not is_artifact:
            for pattern in page_num_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_artifact = True
                    break
        
        # Check generic company name standalone at top/bottom of pages (often happens)
        # Only if it strictly equals the company name or "Company Name Limited"
        if not is_artifact:
            if stripped.lower() == company_name.lower():
                is_artifact = True
        
        if not is_artifact:
            cleaned_lines.append(line) # Keep original line with indentation if any
    
    return '\n'.join(cleaned_lines)

def fix_broken_line_joins(text: str) -> str:
    """
    Fix broken line joins where a sentence is split across lines.
    Preserves double newlines (paragraphs).
    """
    lines = text.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i].strip()
        
        # Preserve paragraph breaks (empty lines)
        if not current_line:
            fixed_lines.append('')
            i += 1
            continue
            
        # If we have a next line
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            
            # If next line is empty, current line ends a block
            if not next_line:
                fixed_lines.append(current_line)
                i += 1
                continue
                
            # Heuristic for joining:
            # Current line does NOT end in punctuation 
            # AND Next line does NOT start with a bullet point or Capital letter (usually)
            # However, sentences often start with Capitals, so we must be careful.
            # Safe bet: Join if current line seems incomplete (no punctuation at end)
            # AND next line is not a purely numeric/list item.
            
            ends_with_punctuation = current_line[-1] in {'.', '!', '?', ':', ';'}
            
            if not ends_with_punctuation:
                # Potential join candidate.
                # Check if next line looks like a new header or list item
                next_is_list = re.match(r'^[\d\-\u2022]+\.', next_line) or next_line.startswith(('-', '*', '•'))
                
                if not next_is_list:
                     # Join them
                    fixed_lines.append(current_line + ' ' + next_line)
                    i += 2
                    continue
        
        # Fallback: just add the line
        fixed_lines.append(current_line)
        i += 1
        
    return '\n'.join(fixed_lines)

def remove_extra_spaces(text: str) -> str:
    """Normalize repeated spaces and newlines."""
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines (more than 2) to exactly 2 to preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def clean_text(text: str, company_name: str) -> str:
    """Pipeline for cleaning text."""
    # 1. Page markers
    text = remove_page_markers(text)
    
    # 2. Headers/Footers
    text = remove_headers_footers(text, company_name)
    
    # 3. Fix broken lines
    text = fix_broken_line_joins(text)
    
    # 4. Global whitespace normalization
    text = remove_extra_spaces(text)
    
    return text.strip()

def process_text_file(input_path: Path, output_dir: Path):
    """Read, clean, and save a single file."""
    print(f"\nProcessing: {input_path.name}")
    
    # Derive company name from filename (e.g. TRENT_2022 -> TRENT)
    # Be robust if filename doesn't have underscore
    stem = input_path.stem
    if '_' in stem:
        company_name = stem.split('_')[0]
    else:
        company_name = stem
        
    print(f"  Company: {company_name}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        original_words = count_words(raw_text)
        print(f"  Original words: {original_words}")
        
        cleaned_text = clean_text(raw_text, company_name)
        cleaned_words = count_words(cleaned_text)
        print(f"  Cleaned words:  {cleaned_words}")
        
        # Use the original filename stem to preserve year (e.g. TRENT_2022.txt)
        output_file = output_dir / f"{stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
        print(f"  Saved to: {output_file}")
        
    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}")

def main():
    # Define directories
    # Using absolute paths or relative to project root is safer
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "Data" / "extracted_text"
    output_dir = base_dir / "Data" / "cleaned_text"
    
    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        return
        
    # Ensure output directory exists (clean it if needed, but overwrite is fine)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files
    files = sorted(list(input_dir.glob("*.txt")))
    
    if not files:
        print("No .txt files found in input directory.")
        return
        
    print(f"Found {len(files)} files to process.")
    
    for file_path in files:
        process_text_file(file_path, output_dir)
        
    print("\n" + "="*30)
    print("All files processed.")

if __name__ == "__main__":
    main()
