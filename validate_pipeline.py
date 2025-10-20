#!/usr/bin/env python3
"""
Pipeline validation utilities for debugging and monitoring.
"""

import os
import pandas as pd
from datasets import load_from_disk

def validate_dataset(dataset_path):
    """Validate dataset structure and content."""
    print(f"🔍 Validating dataset: {dataset_path}")

    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            # Handle Arrow format
            dataset = load_from_disk(dataset_path)
            df = dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()

        print(f"✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        # Check required columns
        required_columns = ['text', 'generated']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        print(f"✅ Required columns present: {required_columns}")

        # Check class balance
        if 'generated' in df.columns:
            class_counts = df['generated'].value_counts()
            total = len(df)
            print(f"📊 Class distribution:")
            for cls, count in class_counts.items():
                label = "Human" if cls == 0 else "AI Generated"
                percentage = (count / total) * 100
                print(f"   {label}: {count} ({percentage:.1f}%)")

        # Check for perplexity
        if 'perplexity' in df.columns:
            ppl_stats = df['perplexity'].describe()
            non_nan = df['perplexity'].notna().sum()
            print(f"🧠 Perplexity available for {non_nan}/{total} samples ({non_nan/total*100:.1f}%)")
            if non_nan > 0:
                print(f"   Range: {ppl_stats['min']:.2f} - {ppl_stats['max']:.2f} (mean: {ppl_stats['mean']:.2f})")
        else:
            print(f"⚠️  No perplexity column found")

        # Check text quality
        text_lengths = df['text'].astype(str).str.len()
        print(f"📝 Text length stats:")
        print(f"   Min: {text_lengths.min()}, Max: {text_lengths.max()}, Mean: {text_lengths.mean():.1f}")
        empty_texts = (df['text'].astype(str).str.strip() == '').sum()
        if empty_texts > 0:
            print(f"⚠️  Found {empty_texts} empty text samples")
        else:
            print(f"✅ No empty text samples found")

        return True

    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False

def validate_features(features_path):
    """Validate features file structure and content."""
    print(f"🔍 Validating features: {features_path}")

    try:
        if not os.path.exists(features_path):
            print(f"❌ Features file not found: {features_path}")
            return False

        df = pd.read_csv(features_path)
        print(f"✅ Features loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        # Check required columns
        if 'generated' not in df.columns:
            print(f"❌ Missing 'generated' column")
            return False

        feature_columns = [col for col in df.columns if col != 'generated']
        print(f"📈 Found {len(feature_columns)} feature columns")

        # Check for perplexity
        if 'perplexity' in feature_columns:
            non_nan = df['perplexity'].notna().sum()
            total = len(df)
            print(f"🧠 Perplexity available for {non_nan}/{total} samples ({non_nan/total*100:.1f}%)")
        else:
            print(f"⚠️  No perplexity feature found")

        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing = [col for col, count in missing_counts.items() if count > len(df) * 0.1]
        if high_missing:
            print(f"⚠️  Features with >10% missing values: {high_missing}")

        return True

    except Exception as e:
        print(f"❌ Error validating features: {e}")
        return False

def validate_pipeline_consistency(dataset_path, features_path):
    """Validate that dataset and features are consistent."""
    print(f"🔍 Validating pipeline consistency...")

    try:
        # Load both
        if dataset_path.endswith('.csv'):
            dataset_df = pd.read_csv(dataset_path)
        else:
            dataset = load_from_disk(dataset_path)
            dataset_df = dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()

        features_df = pd.read_csv(features_path)

        # Check size consistency
        dataset_size = len(dataset_df)
        features_size = len(features_df)

        print(f"📊 Dataset size: {dataset_size}")
        print(f"📈 Features size: {features_size}")

        if dataset_size != features_size:
            print(f"⚠️  Size mismatch between dataset and features")
            print(f"   This may cause issues in the pipeline")
        else:
            print(f"✅ Dataset and features sizes match")

        return True

    except Exception as e:
        print(f"❌ Error validating consistency: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate pipeline components")
    parser.add_argument('--dataset', default='data/datasets/combined_dataset.csv', help='Dataset path')
    parser.add_argument('--features', default='data/features/extracted_features.csv', help='Features path')
    args = parser.parse_args()

    print("🚀 Pipeline Validation Report")
    print("=" * 50)

    # Validate dataset
    dataset_ok = validate_dataset(args.dataset)
    print()

    # Validate features
    features_ok = validate_features(args.features)
    print()

    # Validate consistency
    if dataset_ok and features_ok:
        consistency_ok = validate_pipeline_consistency(args.dataset, args.features)
    else:
        consistency_ok = False

    print()
    print("=" * 50)
    if dataset_ok and features_ok and consistency_ok:
        print("✅ Pipeline validation PASSED")
    else:
        print("❌ Pipeline validation FAILED - check the issues above")