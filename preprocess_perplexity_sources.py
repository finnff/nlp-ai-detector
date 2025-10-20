#!/usr/bin/env python3
"""
Preprocess perplexity for source datasets independently.

This script calculates perplexity for each source dataset and creates enhanced versions
with perplexity values embedded, enabling flexible recombination without reprocessing.
"""

import os
import sys
import math
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob
import json

# GPU detection and setup
def detect_and_setup_gpu():
    """Detect CUDA availability and set up GPU environment."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available. This script requires GPU support.")
            print("Please run on a machine with CUDA-enabled GPU.")
            sys.exit(1)

        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return True, device
    except ImportError:
        print("❌ PyTorch not installed. Installing required dependencies...")
        install_dependencies()
        return detect_and_setup_gpu()  # Retry after installation

def install_dependencies():
    """Install required dependencies for perplexity calculation."""
    CU121_INDEX = "https://download.pytorch.org/whl/cu121"
    REQUIRED_PIP_PKGS = [
        f"torch --index-url {CU121_INDEX}",
        f"torchvision --index-url {CU121_INDEX}",
        f"torchaudio --index-url {CU121_INDEX}",
        "transformers",
        "accelerate",
        "pandas",
        "numpy",
        "tqdm",
    ]

    for spec in REQUIRED_PIP_PKGS:
        print(f"Installing {spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + spec.split())

class PerplexityProcessor:
    """Handles perplexity calculation for datasets."""

    def __init__(self, model_name: str = "EleutherAI/pythia-1.4b", max_len: int = 1536,
                 stride: int = 1536, batch_size: int = 1):
        self.model_name = model_name
        self.max_len = max_len
        self.stride = stride
        self.batch_size = batch_size
        self.device = None
        self.tokenizer = None
        self.model = None

    def setup_model(self):
        """Initialize the language model for perplexity calculation."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        )
        self.model.config.use_cache = False
        if hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "eager"

        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")

    def calculate_perplexity_for_text(self, text: str) -> Tuple[float, int]:
        """Calculate perplexity for a single text."""
        import torch

        if not text or not str(text).strip():
            return float("nan"), 0

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        seq_len = input_ids.size(1)

        if seq_len <= 1:
            return float("nan"), int(seq_len)

        nll_sum = 0.0
        n_tokens = 0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for begin in range(0, seq_len, self.stride):
                end = min(begin + self.max_len, seq_len)
                ids_slice = input_ids[:, begin:end]

                context_len = 0 if begin == 0 else max(0, self.max_len - self.stride)
                context_len = min(context_len, ids_slice.size(1) - 1)

                labels = ids_slice.clone()
                if context_len > 0:
                    labels[:, :context_len] = -100

                out = self.model(input_ids=ids_slice, labels=labels, use_cache=False)
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

    def process_dataset(self, input_path: str, output_path: str, text_column: str = "text", file_type: str = "csv") -> bool:
        """Process a dataset and add perplexity column."""
        import pandas as pd
        from tqdm import tqdm

        print(f"Processing dataset: {input_path}")

        # Load dataset
        if file_type == 'arrow' or input_path.endswith('.arrow'):
            from datasets import load_from_disk
            try:
                dataset = load_from_disk(input_path)
                if 'train' in dataset:
                    df = dataset['train'].to_pandas()
                    print(f"✅ Loaded Arrow dataset with train split: {len(df)} samples")
                else:
                    df = dataset.to_pandas()
                    print(f"✅ Loaded Arrow dataset: {len(df)} samples")
            except Exception as e:
                print(f"❌ Error loading Arrow dataset: {e}")
                return False
        elif input_path.endswith('.xlsx') or input_path.endswith('.xls'):
            # Try multiple approaches for Excel files
            df = None

            # Try different engines and approaches
            approaches = [
                {'engine': 'openpyxl', 'description': 'openpyxl engine'},
                {'engine': 'xlrd', 'description': 'xlrd engine'},
            ]

            for approach in approaches:
                try:
                    print(f"📊 Trying to read Excel file with {approach['description']}")
                    df = pd.read_excel(input_path, engine=approach['engine'])

                    # Check if we can access text column
                    if text_column not in df.columns:
                        print(f"⚠️  Text column '{text_column}' not found. Available columns: {list(df.columns)}")
                        # Try common alternative column names
                        alt_columns = ['text', 'Text', 'content', 'Content', 'essay', 'Essay']
                        for alt_col in alt_columns:
                            if alt_col in df.columns:
                                print(f"🔄 Using alternative column: {alt_col}")
                                text_column = alt_col
                                break
                        else:
                            print(f"❌ No suitable text column found")
                            return False

                    # Test a few text values to ensure encoding worked
                    test_texts = df[text_column].astype(str).head(5)
                    print(f"✅ Successfully read Excel file with {approach['description']}")
                    print(f"📝 Sample text preview: {test_texts.iloc[0][:100]}...")
                    break

                except Exception as e:
                    print(f"❌ Failed with {approach['description']}: {e}")
                    continue

            # If all engines fail, try basic approach without specifying engine
            if df is None:
                try:
                    print("📊 Trying basic pandas read_excel...")
                    df = pd.read_excel(input_path)
                    if text_column in df.columns:
                        print(f"✅ Successfully read Excel file with default approach")
                    else:
                        print(f"⚠️  Text column '{text_column}' not found in default approach")
                        return False
                except Exception as e:
                    print(f"❌ All Excel reading approaches failed: {e}")
                    return False

            if df is None:
                print(f"❌ Could not read Excel file with any approach")
                return False
        else:
            # CSV files - try multiple encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(input_path, encoding=encoding)
                    if text_column in df.columns:
                        print(f"✅ Successfully read CSV file with encoding: {encoding}")
                        break
                except Exception as e:
                    print(f"❌ Failed with encoding {encoding}: {e}")
                    continue

            if df is None:
                print(f"❌ Could not read CSV file with any encoding")
                return False

        print(f"Loaded {len(df)} samples")

        if text_column not in df.columns:
            print(f"❌ Text column '{text_column}' not found in {input_path}")
            return False

        # Calculate perplexity for each text
        perplexities = []
        tokens_count = []

        for text in tqdm(df[text_column], desc="Calculating perplexity"):
            ppl, tokens = self.calculate_perplexity_for_text(str(text))
            perplexities.append(ppl)
            tokens_count.append(tokens)

        # Add perplexity columns
        df['perplexity'] = perplexities
        df['ppl_tokens'] = tokens_count
        df['ppl_model'] = self.model_name

        # Save enhanced dataset
        if file_type == 'arrow' or output_path.endswith('.arrow'):
            from datasets import Dataset
            try:
                print(f"💾 Saving Arrow dataset to: {output_path}")
                Dataset.from_pandas(df).save_to_disk(output_path)
                print(f"✅ Saved Arrow dataset successfully")
            except Exception as e:
                print(f"❌ Error saving Arrow dataset: {e}")
                return False
        else:
            df.to_csv(output_path, index=False)
            print(f"✅ Saved CSV dataset to: {output_path}")

        print(f"Perplexity stats: min={df['perplexity'].min():.2f}, max={df['perplexity'].max():.2f}, mean={df['perplexity'].mean():.2f}")

        return True

def detect_source_datasets(datasets_dir: str) -> Dict[str, Dict]:
    """Detect available source datasets and their configurations."""
    sources = {}
    print(f"🔍 Scanning datasets directory: {datasets_dir}")

    # Debug: list all directories and files in datasets_dir
    try:
        all_items = os.listdir(datasets_dir)
        print(f"📁 Found {len(all_items)} items in datasets directory:")
        for item in sorted(all_items):
            item_path = os.path.join(datasets_dir, item)
            if os.path.isdir(item_path):
                print(f"  📂 {item}/ (directory)")
                # List contents of subdirectory
                try:
                    sub_items = os.listdir(item_path)[:5]  # Show first 5 items
                    for sub_item in sub_items:
                        print(f"    - {sub_item}")
                    if len(os.listdir(item_path)) > 5:
                        print(f"    ... and {len(os.listdir(item_path)) - 5} more items")
                except Exception as e:
                    print(f"    ❌ Could not list contents: {e}")
            else:
                print(f"  📄 {item} (file)")
    except Exception as e:
        print(f"❌ Error scanning datasets directory: {e}")
        return sources

    # Check for Arrow datasets (directories, not files)
    arrow_sources = {
        'ai_text_detection_pile': 'dataset.arrow',  # These are actually directories
        'hc3': 'dataset.arrow',
    }

    for source_name, expected_name in arrow_sources.items():
        source_path = os.path.join(datasets_dir, source_name)

        # Arrow datasets are saved as directories by datasets.save_to_disk()
        # Check if it's a directory that looks like an Arrow dataset
        if os.path.isdir(source_path):
            # Look for indicators of Arrow dataset structure
            dataset_files = ['dataset_info.json', 'dataset_dict.json', 'state.json']
            arrow_indicators = ['data-00000-of-00001.arrow', 'dataset.arrow', 'features.json']

            has_dataset_structure = any(
                os.path.exists(os.path.join(source_path, f))
                for f in dataset_files + arrow_indicators
            )

            if has_dataset_structure:
                print(f"📊 Found Arrow dataset: {source_name} at {source_path}")
                sources[source_name] = {
                    'type': 'arrow',
                    'input_path': source_path,  # Directory path, not file
                    'output_path': os.path.join(source_path, 'dataset_with_perplexity'),
                    'text_column': 'text'
                }
            else:
                print(f"⚠️  Directory exists but doesn't look like Arrow dataset: {source_path}")
        else:
            print(f"📁 Directory not found: {source_path}")

    # Check for CSV datasets
    csv_sources = {
        'daigt_v2': 'DAIGT_v2_train_v2_drcat_02.csv',
        'sunilthite': 'sunilthite_Training_Essay_Data.csv',
        'llm_detect_competition': 'kaggleComp_train_essays.csv',
    }

    for source_name, filename in csv_sources.items():
        file_path = os.path.join(datasets_dir, source_name, filename)
        if os.path.exists(file_path):
            sources[source_name] = {
                'type': 'csv',
                'input_path': file_path,
                'output_path': file_path.replace('.csv', '_with_perplexity.csv'),
                'text_column': 'text'
            }

    # Check for Excel datasets
    if os.path.exists(os.path.join(datasets_dir, 'ah_aitd', 'AHAIRD_Dataset.xlsx')):
        sources['ah_aitd'] = {
            'type': 'excel',
            'input_path': os.path.join(datasets_dir, 'ah_aitd', 'AHAIRD_Dataset.xlsx'),
            'output_path': os.path.join(datasets_dir, 'ah_aitd', 'AHAIRD_Dataset_with_perplexity.csv'),
            'text_column': 'text'
        }

    return sources

def main():
    parser = argparse.ArgumentParser(description="Preprocess perplexity for source datasets")
    parser.add_argument('--sources', nargs='*', help='Specific sources to process (default: all)')
    parser.add_argument('--all', action='store_true', help='Process all available sources')
    parser.add_argument('--model', help='Model for perplexity calculation (overrides config)')
    parser.add_argument('--max_len', type=int, help='Maximum sequence length (overrides config)')
    parser.add_argument('--stride', type=int, help='Stride for sliding window (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config, not implemented yet)')
    parser.add_argument('--resume', action='store_true', help='Resume processing (skip existing outputs)')
    parser.add_argument('--datasets_dir', default='data/datasets', help='Datasets directory')
    parser.add_argument('--config', default='dataset_configuration.toml', help='Configuration file')

    args = parser.parse_args()

    # Load configuration
    try:
        import tomllib
        with open(args.config, 'rb') as f:
            config = tomllib.load(f)

        # Extract perplexity config with CLI overrides
        perplexity_config = config.get('perplexity', {})
        model = args.model or perplexity_config.get('model', 'EleutherAI/pythia-1.4b')
        max_len = args.max_len or perplexity_config.get('max_len', 1536)
        stride = args.stride or perplexity_config.get('stride', 1536)
        batch_size = args.batch_size or perplexity_config.get('batch_size', 1)
        resume = args.resume or perplexity_config.get('resume', False)

        print(f"✅ Loaded configuration from {args.config}")
        print(f"  Model: {model}")
        print(f"  Max length: {max_len}")
        print(f"  Stride: {stride}")
        print(f"  Resume: {resume}")

    except Exception as e:
        print(f"⚠️  Could not load config file: {e}")
        print("Using default values...")
        model = args.model or 'EleutherAI/pythia-1.4b'
        max_len = args.max_len or 1536
        stride = args.stride or 1536
        batch_size = args.batch_size or 1
        resume = args.resume

    if not args.all and not args.sources:
        print("❌ Please specify either --all or --sources <source1> <source2> ...")
        return 1

    # Setup GPU
    gpu_available, device = detect_and_setup_gpu()
    if not gpu_available:
        return 1

    # Detect source datasets
    sources = detect_source_datasets(args.datasets_dir)
    print(f"Detected {len(sources)} source datasets:")
    for name, config in sources.items():
        print(f"  - {name}: {config['type']} ({config['input_path']})")

    # Filter sources to process
    if args.sources:
        sources_to_process = {k: v for k, v in sources.items() if k in args.sources}
        missing = set(args.sources) - set(sources.keys())
        if missing:
            print(f"❌ Sources not found: {missing}")
            return 1
    else:
        sources_to_process = sources

    if not sources_to_process:
        print("❌ No sources to process")
        return 1

    print(f"\nProcessing {len(sources_to_process)} sources...")

    # Initialize perplexity processor
    processor = PerplexityProcessor(
        model_name=model,
        max_len=max_len,
        stride=stride,
        batch_size=batch_size
    )
    processor.device = device
    processor.setup_model()

    # Process each source
    successful = []
    failed = []

    for source_name, config in sources_to_process.items():
        print(f"\n{'='*60}")
        print(f"Processing source: {source_name}")
        print(f"{'='*60}")

        # Skip if output exists and resume is enabled
        if resume and os.path.exists(config['output_path']):
            print(f"⏭️  Skipping {source_name} (output already exists)")
            successful.append(source_name)
            continue

        try:
            success = processor.process_dataset(
                input_path=config['input_path'],
                output_path=config['output_path'],
                text_column=config['text_column'],
                file_type=config['type']
            )

            if success:
                successful.append(source_name)
                print(f"✅ Completed {source_name}")
            else:
                failed.append(source_name)
                print(f"❌ Failed {source_name}")

        except Exception as e:
            print(f"❌ Error processing {source_name}: {e}")
            failed.append(source_name)

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)} sources")
    for source in successful:
        print(f"  - {source}")

    if failed:
        print(f"❌ Failed: {len(failed)} sources")
        for source in failed:
            print(f"  - {source}")

    print(f"\n🎉 Perplexity preprocessing complete!")
    print("You can now use combine_dataset.py as usual - it will automatically")
    print("use the perplexity-enhanced versions when available.")

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())