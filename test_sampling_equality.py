#!/usr/bin/env python3
"""
Test script to verify if HuggingFace and pandas sampling methods produce identical results.

This will help us understand why XGBoost performs poorly with limited datasets.
"""

import pandas as pd
import numpy as np
from datasets import load_from_disk, Dataset
import os

def test_sampling_equality():
    print("🧪 Testing Sampling Method Equality")
    print("=" * 60)

    # Load the combined dataset
    dataset_path = 'data/datasets/combined_dataset.csv'
    if not os.path.exists(dataset_path):
        print("❌ Combined dataset not found. Please run combine_dataset.py first.")
        return False

    print(f"📁 Loading dataset from {dataset_path}")

    # Method 1: Load as pandas DataFrame (like our XGBoost approach)
    df = pd.read_csv(dataset_path)
    print(f"📊 Loaded DataFrame: {df.shape} samples")

    # Method 2: Load as HuggingFace Dataset (like main pipeline)
    ds = Dataset.from_pandas(df)
    print(f"📊 Loaded Dataset: {len(ds)} samples")

    # Test parameters (same as configuration.toml)
    random_state = 42
    num_samples = 1000  # Test with a sample size that shows the issue
    test_size = 0.2

    print(f"\n🔧 Testing with:")
    print(f"   Random state: {random_state}")
    print(f"   Sample limit: {num_samples}")
    print(f"   Test size: {test_size}")

    # Method 1: HuggingFace sampling (main pipeline)
    print(f"\n📋 Method 1: HuggingFace Dataset")
    print("-" * 40)

    ds_hf = ds.shuffle(seed=random_state)
    ds_limited_hf = ds_hf.select(range(min(num_samples, len(ds_hf))))

    # Get text samples for comparison
    hf_texts = [sample['text'] for sample in ds_limited_hf]
    hf_labels = [sample['generated'] for sample in ds_limited_hf]

    print(f"   After shuffle + limit: {len(hf_texts)} samples")
    print(f"   First 3 texts (HuggingFace):")
    for i, text in enumerate(hf_texts[:3]):
        print(f"     {i+1}. {text[:50]}...")

    # Method 2: Pandas sampling (our XGBoost approach)
    print(f"\n📋 Method 2: Pandas DataFrame")
    print("-" * 40)

    # Apply same transformations as our XGBoost code
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)

    # Apply limiting
    df_limited = df_shuffled.iloc[:num_samples]

    pd_texts = df_limited['text'].tolist()
    pd_labels = df_limited['generated'].tolist()

    print(f"   After shuffle + limit: {len(pd_texts)} samples")
    print(f"   First 3 texts (Pandas):")
    for i, text in enumerate(pd_texts[:3]):
        print(f"     {i+1}. {text[:50]}...")

    # Compare results
    print(f"\n🔍 Comparison Results")
    print("=" * 40)

    # Check if lengths match
    if len(hf_texts) != len(pd_texts):
        print(f"❌ Length mismatch: HF={len(hf_texts)}, PD={len(pd_texts)}")
        return False

    # Check if texts match
    matching_texts = 0
    mismatched_indices = []

    for i, (hf_text, pd_text) in enumerate(zip(hf_texts, pd_texts)):
        if hf_text == pd_text:
            matching_texts += 1
        else:
            mismatched_indices.append(i)

    print(f"✅ Matching texts: {matching_texts}/{len(hf_texts)} ({matching_texts/len(hf_texts)*100:.1f}%)")

    if mismatched_indices:
        print(f"❌ Mismatched at indices: {mismatched_indices[:10]}... (showing first 10)")

        # Show a few examples of mismatches
        print(f"\n📝 Examples of mismatches:")
        for i in mismatched_indices[:3]:
            print(f"   Index {i}:")
            print(f"     HF: {hf_texts[i][:80]}...")
            print(f"     PD: {pd_texts[i][:80]}...")
            print()

    # Check if labels match
    matching_labels = sum(1 for hf_label, pd_label in zip(hf_labels, pd_labels) if hf_label == pd_label)
    print(f"✅ Matching labels: {matching_labels}/{len(hf_labels)} ({matching_labels/len(hf_labels)*100:.1f}%)")

    # Final verdict
    if matching_texts == len(hf_texts) and matching_labels == len(hf_labels):
        print(f"\n🎉 SUCCESS: Sampling methods are IDENTICAL!")
        print("   XGBoost and main pipeline should produce identical results.")
        return True
    else:
        print(f"\n❌ FAILURE: Sampling methods are DIFFERENT!")
        print("   This explains why XGBoost performs poorly with limited datasets.")
        print("   XGBoost trains on different samples than what's being evaluated.")
        return False

def test_train_test_split():
    """Test if train_test_split produces identical results."""
    print(f"\n🧪 Testing Train/Test Split Equality")
    print("=" * 60)

    # Load dataset
    dataset_path = 'data/datasets/combined_dataset.csv'
    df = pd.read_csv(dataset_path)

    # Limit to test size
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
    df_limited = df_shuffled.iloc[:1000]

    X = df_limited.drop('generated', axis=1)
    y = df_limited['generated']

    print(f"📊 Features shape: {X.shape}")
    print(f"📊 Labels shape: {y.shape}")

    # Test with same parameters
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"✅ Split 1: Train={len(X_train1)}, Test={len(X_test1)}")
    print(f"✅ Split 2: Train={len(X_train2)}, Test={len(X_test2)}")

    # Compare results
    train_match = X_train1.equals(X_train2)
    test_match = X_test1.equals(X_test2)
    y_train_match = y_train1.equals(y_train2)
    y_test_match = y_test1.equals(y_test2)

    print(f"\n🔍 Split Comparison:")
    print(f"   X_train match: {train_match}")
    print(f"   X_test match: {test_match}")
    print(f"   y_train match: {y_train_match}")
    print(f"   y_test match: {y_test_match}")

    if all([train_match, test_match, y_train_match, y_test_match]):
        print(f"🎉 SUCCESS: Train/test splits are IDENTICAL!")
        return True
    else:
        print(f"❌ FAILURE: Train/test splits are DIFFERENT!")
        return False

if __name__ == "__main__":
    # Import train_test_split
    from sklearn.model_selection import train_test_split

    print("🚀 Testing Sampling Method Equality")
    print("This will help identify why XGBoost performs poorly with limited datasets.\n")

    # Test 1: Sampling equality
    sampling_success = test_sampling_equality()

    # Test 2: Split consistency
    split_success = test_train_test_split()

    # Summary
    print(f"\n🏁 FINAL RESULTS")
    print("=" * 60)
    print(f"Sampling methods identical: {'✅ YES' if sampling_success else '❌ NO'}")
    print(f"Train/test splits identical: {'✅ YES' if split_success else '❌ NO'}")

    if sampling_success and split_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   XGBoost should work correctly regardless of sample size.")
    else:
        print(f"\n⚠️  TESTS FAILED!")
        print("   This explains the XGBoost performance issues.")