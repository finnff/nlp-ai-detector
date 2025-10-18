"""I ran this script on the university jupyter server. Works there, definitely not on my M1, not sure about FInn's setup."""

import sys
import os
import math
import subprocess
import importlib
import argparse
from datetime import datetime
from typing import Tuple

# ------------------------------ installs ------------------------------------- #
CU121_INDEX = "https://download.pytorch.org/whl/cu121"

REQUIRED_PIP_PKGS = [
    # Pin to CUDA 12.1 wheels for PyTorch stack
    f"torch --index-url {CU121_INDEX}",
    f"torchvision --index-url {CU121_INDEX}",
    f"torchaudio --index-url {CU121_INDEX}",
    # Rest can come from default index
    "transformers",
    "accelerate",
    "pandas",
    "numpy",
    "tqdm",
]

def _pip_install(spec: str) -> None:
    print(f"[install] {spec}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + spec.split())

def ensure_deps() -> None:
    # Try a minimal import to see if torch-cuda is already fine
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import tqdm  # noqa: F401
        return
    except Exception:
        pass
    for spec in REQUIRED_PIP_PKGS:
        _pip_install(spec)

ensure_deps()

# ------------------------------ imports -------------------------------------- #
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------ core ----------------------------------------- #
def prepare_device() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not visible. This script is for CUDA servers only. Fix your environment.")
    device = "cuda"
    # Ampere-friendly perf flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    return device

def load_data(path: str, samples: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index().rename(columns={"index": "orig_index"})
    if "text" not in df.columns:
        raise ValueError("CSV needs a 'text' column.")
    df["word_count"] = df["text"].astype(str).apply(lambda s: len(s.split()))
    if samples is not None and samples >= 0:
        df = df.sample(n=samples, random_state=seed).sort_index()
    print(f"[data] total rows: {len(df)}")
    return df

def load_model(model_name: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    model.to(device)
    model.eval()
    print(f"[model] {model_name} loaded on {torch.cuda.get_device_name(0)}")
    return tokenizer, model

@torch.inference_mode()
def perplexity_for_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_len: int,
    stride: int,
    doc_token_cap: int | None,
) -> Tuple[float, int]:
    if not text or not str(text).strip():
        return float("nan"), 0

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    if doc_token_cap is not None:
        input_ids = input_ids[:, :doc_token_cap]

    seq_len = input_ids.size(1)
    if seq_len <= 1:
        return float("nan"), int(seq_len)

    nll_sum = 0.0
    n_tokens = 0

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            ids_slice = input_ids[:, begin:end]

            context_len = 0 if begin == 0 else max(0, max_len - stride)
            context_len = min(context_len, ids_slice.size(1) - 1)

            labels = ids_slice.clone()
            if context_len > 0:
                labels[:, :context_len] = -100

            out = model(input_ids=ids_slice, labels=labels, use_cache=False)
            valid = (labels != -100).sum().item()
            if valid > 0:
                nll_sum += out.loss.item() * valid
                n_tokens += valid

            del ids_slice, labels, out
            torch.cuda.empty_cache()

            if end == seq_len:
                break

    if n_tokens == 0:
        return float("inf"), 0

    ppl = math.exp(nll_sum / n_tokens)
    return float(ppl), int(n_tokens)

# ------------------------------ main function -------------------------------- #
def calculate_perplexity(
    dataset_path: str,
    output_path: str,
    model_name: str = "EleutherAI/pythia-1.4b",
    max_len: int = 1536,
    stride: int = 1536,
    samples: int | None = None,
    seed: int = 42,
    doc_token_cap: int | None = None,
) -> str:
    """w
    Calculate perplexity for texts in a dataset.
    
    Args:
        dataset_path: Path to input CSV with a 'text' column
        output_path: Base path for output CSV (timestamp will be added)
        model_name: HuggingFace model ID for perplexity calculation
        max_len: Sliding window length in tokens
        stride: Stride in tokens (use same as max_len to avoid overlap)
        samples: Number of samples to process (None for all)
        seed: Random seed for sampling
        doc_token_cap: Optional cap on tokens per document
    
    Returns:
        str: Path to the output file with timestamp
    """
    # Generate timestamp and modify output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path, ext = os.path.splitext(output_path)
    timestamped_output = f"{base_path}_{timestamp}{ext}"
    
    device = prepare_device()
    df = load_data(dataset_path, samples, seed)
    tokenizer, model = load_model(model_name, device)

    results = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Perplexity"):
        text = getattr(row, "text")
        idx = getattr(row, "orig_index")
        ppl, toks = perplexity_for_text(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_len=max_len,
            stride=stride,
            doc_token_cap=doc_token_cap,
        )
        results.append(
            {
                "orig_index": int(idx),
                "perplexity": ppl,
                "ppl_tokens": toks,
                "model": model_name,
            }
        )

    res_df = pd.DataFrame(results).sort_values("orig_index").reset_index(drop=True)
    outdir = os.path.dirname(timestamped_output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    res_df.to_csv(timestamped_output, index=False)
    print(f"[done] wrote {len(res_df)} rows to {timestamped_output}")

    # quick stats
    print("\n[stats] perplexity describe:")
    print(res_df["perplexity"].describe())
    
    return timestamped_output

# ------------------------------ CLI ------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute perplexity over a CSV using a CUDA model.")
    p.add_argument("--dataset", default="combined_dataset.csv", help="Path to input CSV with a 'text' column.")
    p.add_argument("--output", default="res/perplexity_results.csv", help="Output CSV path.")
    p.add_argument("--model", default="EleutherAI/pythia-1.4b", help="HF model id for perplexity.")
    p.add_argument("--max_len", type=int, default=1536, help="Sliding window length (tokens).")
    p.add_argument("--stride", type=int, default=1536, help="Stride (tokens). Use same as max_len to avoid overlap.")
    p.add_argument("--samples", type=int, default=None, help="If set, sample this many rows for a test run. Use -1 for all.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument("--doc_token_cap", type=int, default=None, help="Optional cap on tokens per doc for scoring.")
    return p.parse_args()

def main() -> str:
    """Main function for CLI usage."""
    args = parse_args()
    
    output_file = calculate_perplexity(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model,
        max_len=args.max_len,
        stride=args.stride,
        samples=args.samples if args.samples != -1 else None,
        seed=args.seed,
        doc_token_cap=args.doc_token_cap,
    )
    
    return output_file

if __name__ == "__main__":
    # lazy imports used above
    import pandas as pd  # noqa: F401
    output_file = main()
    print(f"\n[output] Results saved to: {output_file}")
