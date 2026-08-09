import sys
import pandas as pd
from llama_cpp import Llama

# ==================== User Configuration Area (modify the following variables directly) ====================

# GGUF model file path
MODEL_PATH = "F:/modelscope_cache/models/Finetuned_model/Ministral-3-8B-Instruct-2512-science-finetuned.gguf"

# CSV data file path
CSV_PATH = "./data.csv"

# Prompt: can be a file path or a direct string
# If a file path is provided, ensure the file exists; otherwise it will be used as the prompt string
# The prompt must contain the {data} placeholder for inserting CSV data
PROMPT = "./prompt_template.txt"  # or directly write a string, e.g.: "Please analyze the following data:\n{data}\nAnalysis result:"

# Model context length (adjust according to model and CSV data size)
N_CTX = 8192

# Generation temperature
TEMPERATURE = 0.7

# Maximum number of tokens to generate
MAX_TOKENS = 4096

# Whether to print verbose logs
VERBOSE = True

# Maximum number of CSV rows to display (to avoid exceeding context length)
MAX_CSV_ROWS = 60

# GPU acceleration settings
# Number of model layers to offload to GPU (-1 means all layers, 0 means CPU only)
# If you encounter GPU memory shortage, reduce this value appropriately (e.g., 20, 30)
N_GPU_LAYERS = -1

# ==================== Functional code below, generally no need to modify ====================

def csv_to_markdown(csv_path: str, max_rows: int = MAX_CSV_ROWS) -> str:
    """Read CSV file and convert it to a Markdown table string."""
    df = pd.read_csv(csv_path)
    if len(df) > max_rows:
        print(f"Warning: CSV file contains {len(df)} rows, only displaying the first {max_rows} rows to avoid exceeding context length.")
        df = df.head(max_rows)
    return df.to_markdown(index=False)

def load_prompt(prompt_source: str) -> str:
    """
    Load prompt.
    If prompt_source is a file path, read the file content;
    otherwise return the prompt_source string directly.
    """
    try:
        # Try to read as a file path
        with open(prompt_source, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Not a file path, return as a string directly
        return prompt_source

def main():
    # 1. Load model
    print("Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        verbose=VERBOSE
    )
    print("Model loading completed.")

    # 2. Read CSV and convert to Markdown table
    print("Reading CSV file...")
    csv_markdown = csv_to_markdown(CSV_PATH)
    if VERBOSE:
        print("CSV content (Markdown):")
        print(csv_markdown)

    # 3. Load prompt template
    prompt_template = load_prompt(PROMPT)
    if "{data}" not in prompt_template:
        print("Warning: {data} placeholder not found in prompt, CSV data will be appended to the end of the prompt.")
        prompt_template += "\n{data}"

    # 4. Construct final prompt
    final_prompt = prompt_template.format(data=csv_markdown)
    if VERBOSE:
        print("Final prompt:")
        print(final_prompt)

    # 5. Call model to generate analysis results
    print("\nGenerating analysis results...\n")
    output = llm(
        final_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True
    )
    for chunk in output:
        text = chunk['choices'][0]['text']
        print(text, end='', flush=True)
    print()

if __name__ == "__main__":
    main()
