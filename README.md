# Financial Analysis Pipeline

A production-ready system for analyzing financial annual reports using PDF extraction, text cleaning, LLM processing, and sentiment analysis.

## Features

- **PDF Extraction**: Extract text from PDF annual reports
- **Text Cleaning**: Clean and chunk text for processing
- **LLM Processing**: Extract structured insights (positive, negative, guidance, risk statements)
- **Sentiment Analysis**: Analyze sentiment using FinBERT model
- **Unified Dataset**: Combine all results into a single dataset with Company, Statement, Category, Sentiment, and Confidence

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
```bash
export GEMINI_API_KEY="your-api-key-here"  # For LLM processing
```

## Configuration

All configuration is centralized in `config.py`. Key settings:

- **Directories**: All input/output directories
- **PDF Extraction**: Input/output paths for PDF processing
- **Text Cleaning**: Chunk size, overlap settings
- **LLM Processing**: Model name, delays, retry settings
- **Sentiment Analysis**: Model name, input/output paths
- **Result Combination**: Output directory and filename

## Usage

### Run Complete Pipeline

```bash
python main.py
```

This will:
1. Extract text from PDFs (if enabled)
2. Clean and chunk text (if enabled)
3. Process through LLM (if enabled and API key available)
4. Run sentiment analysis
5. Combine all results into unified dataset

### Run Individual Steps

You can also run individual scripts:

```bash
# PDF Extraction
python Scripts/pdf\ extraction.py

# Text Cleaning
python Scripts/text\ cleaning.py

# LLM Processing (requires API key)
python Scripts/llm\ processing.py

# Sentiment Analysis
python Scripts/finbert\ sentiment.py

# Combine Results
python Scripts/combine_results.py
```

## Output Structure

### Unified Dataset (`Data/Combined Output/unified_dataset.csv`)

The final output contains:
- **Company**: Company name
- **Statement**: Extracted financial statement
- **Category**: Category (positive, negative, guidance, risk)
- **Sentiment**: Sentiment label (Positive, Negative, Neutral)
- **Confidence**: Confidence score (0.0 to 1.0)

### Intermediate Outputs

- `Data/Extracted Text/`: Raw text extracted from PDFs
- `Data/Cleaned Text/`: Cleaned and chunked text (JSON)
- `Data/LLM Outputs/`: Structured insights from LLM (JSON)
- `Data/Sentiment Output/`: Sentiment analysis results (CSV)
- `Data/Combined Output/`: Final unified dataset (CSV)

## Code Quality Features

✅ **No Hard-coded Paths**: All paths configured in `config.py`  
✅ **Reusable Functions**: Functions are modular and reusable  
✅ **Configurable Parameters**: All parameters can be adjusted in config  
✅ **Reproducible**: Same output for same input  
✅ **Error Handling**: Graceful handling of missing files/errors  

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Notes

- LLM processing requires a Gemini API key (set via `GEMINI_API_KEY` environment variable)
- Sentiment analysis uses the ProsusAI/finbert model (downloaded automatically)
- The system handles missing files gracefully and provides informative error messages





