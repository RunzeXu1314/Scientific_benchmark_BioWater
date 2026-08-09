# deepseek_python_20260628_7b5d13_llamacpp.py
import os
import argparse
import numpy as np
import pandas as pd

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("please install llama-cpp-python: pip install llama-cpp-python")

from utils import data_utils
from utils import general_utils


def compute_prompt_ppl_via_completions(model: Llama, prompt: str) -> float:

    response = model.create_completion(
        prompt=prompt,
        max_tokens=1,
        echo=True,
        logprobs=1,
        temperature=0.0,
        stop=None,
    )

    if not response or "choices" not in response:
        raise RuntimeError("llama.cpp returned empty response")

    choice = response["choices"][0]
    logprobs_data = choice.get("logprobs")
    if not logprobs_data:
        raise ValueError("Response missing logprobs")

    token_logprobs = logprobs_data.get("token_logprobs")
    if not token_logprobs or len(token_logprobs) < 2:
        raise ValueError(f"token_logprobs is empty or too short: {token_logprobs}")

    # Because echo=True, token_logprobs contains all tokens from input and generation
    # The last token is generated, all preceding tokens are the input prompt
    input_logprobs = token_logprobs[:-1]

    # Filter out possible None values (usually not present)
    valid_lps = [lp for lp in input_logprobs if lp is not None]
    if not valid_lps:
        raise ValueError("Unable to extract valid input token logprobs")

    nll = -np.mean(valid_lps)
    ppl = np.exp(nll)
    return ppl


@general_utils.timer
def main(model_path: str, abstracts_fpath: str):
    np.random.seed(42)

    # Extract base name from model path (used as template filename and results directory)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    safe_name = base_name.replace('/', '--').replace(':', '--')

    # Load llama.cpp model, key: logits_all=True
    print(f"Loading model: {model_path}")
    model = Llama(
        model_path=model_path,
        n_ctx=2048,            # Adjust context length as needed
        verbose=False,
        n_gpu_layers=-1,       # Use GPU if available (set to 0 for CPU only)
        logits_all=True,       # Must be enabled to obtain token logprobs
    )

    # Read prompt template
    prompt_template = data_utils.read_prompt_template(safe_name)

    # Read data
    df = pd.read_csv(abstracts_fpath)
    records = []

    for abstract_index, abstract in enumerate(df["combined_abstract"]):
        original_abstract, incorrect_abstract = data_utils.extract_abstract_pair(abstract)

        # Randomly swap order
        if np.random.rand() > 0.5:
            original_abstract, incorrect_abstract = incorrect_abstract, original_abstract
            choice_true = "B"
        else:
            choice_true = "A"

        choices = data_utils.prepare_prompt_multiple_choice_harness(
            original_abstract, incorrect_abstract, prompt_template,
        )
        prompt_A, prompt_B = choices[0], choices[1]

        print(f"-" * 70 + f"\n*** Abstract index: {abstract_index} ***")

        # Compute perplexity for both prompts
        ppl_A = compute_prompt_ppl_via_completions(model, prompt_A)
        ppl_B = compute_prompt_ppl_via_completions(model, prompt_B)

        true_label = 0 if choice_true == "A" else 1

        if ppl_A < ppl_B:
            pred_label = 0
        elif ppl_A > ppl_B:
            pred_label = 1
        else:
            pred_label = -1      # Tie

        is_tie = (pred_label == -1)
        records.append({
            'index': abstract_index,
            'prompt_A': prompt_A,
            'prompt_B': prompt_B,
            'ppl_A': ppl_A,
            'ppl_B': ppl_B,
            'true_label': true_label,
            'pred_label': pred_label,
            'is_tie': is_tie
        })

    # Compute statistics
    df_results = pd.DataFrame(records)
    ties = df_results['is_tie'].sum()
    acc = (df_results['pred_label'] == df_results['true_label']).mean()

    print(f"Number of ties: {ties}")
    print(f"Accuracy: {acc:.4f}")

    # Save results
    results_dir = f"model_results/{safe_name}"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./Abstractdata.csv",
                        help="Path to input CSV file containing abstracts")
    args = parser.parse_args()

    # Please modify to your .gguf model path
    llms = [
        "F:/modelscope_cache/models/Finetuned_model/Ministral-3-8B-Instruct-2512-BF16.gguf",   # Example path
    ]

    for model_path in llms:
        main(model_path, args.input_file)
