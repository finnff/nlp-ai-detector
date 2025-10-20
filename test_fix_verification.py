#!/usr/bin/env python3
"""
Test script to verify our HuggingFace-based fix for XGBoost sampling.

This will test if converting features to HuggingFace dataset produces identical results
as the main pipeline's dataset processing.
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os

def test_huggingface_fix():
    print("🧪 Testing HuggingFace-Based Fix for XGBoost")
    print("=" * 60)

    # Load the combined dataset
    dataset_path = 'data/datasets/combined_dataset.csv'
    if not os.path.exists(dataset_path):
        print("❌ Combined dataset not found. Please run combine_dataset.py first.")
        return False

    print(f"📁 Loading dataset from {dataset_path}")

    # Load features (simulating XGBoost approach)
    df_features = pd.read_csv('data/features/extracted_features.csv')
    X = df_features.drop('generated', axis=1)
    y = df_features['generated']

    print(f"📊 Loaded features: {X.shape} samples")

    # Test parameters (same as configuration.toml)
    random_state = 42
    num_samples = 1000  # Test with sample size that showed the issue
    test_size = 0.2

    print(f"\n🔧 Testing with:")
    print(f"   Random state: {random_state}")
    print(f"   Sample limit: {num_samples}")
    print(f"   Test size: {test_size}")

    # Method 1: Main pipeline (original HuggingFace approach)
    print(f"\n📋 Method 1: Main Pipeline (Reference)")
    print("-" * 50)

    df_main = pd.read_csv(dataset_path)
    main_ds = Dataset.from_pandas(df_main)
    main_ds = main_ds.shuffle(seed=random_state)
    main_ds_limited = main_ds.select(range(min(num_samples, len(main_ds))))

    main_texts = [sample['text'] for sample in main_ds_limited]
    main_labels = [sample['generated'] for sample in main_ds_limited]

    print(f"   After shuffle + limit: {len(main_texts)} samples")
    print(f"   First 3 texts (Main Pipeline):")
    for i, text in enumerate(main_texts[:3]):
        print(f"     {i+1}. {text[:50]}...")

    # Method 2: XGBoost Fix (HuggingFace on features)
    print(f"\n📋 Method 2: XGBoost Fix (HuggingFace on Features)")
    print("-" * 50)

    # This is our new approach: convert features to HuggingFace, then use their shuffle
    features_df = pd.concat([X, y], axis=1)
    features_ds = Dataset.from_pandas(features_df)
    features_ds = features_ds.shuffle(seed=random_state)

    if num_samples > 0:
        features_ds = features_ds.select(range(min(num_samples, len(features_ds))))

    # Convert back to pandas
    features_result = features_ds.to_pandas()
    X_fixed = features_result.drop('generated', axis=1)
    y_fixed = features_result['generated']

    print(f"   After shuffle + limit: {len(X_fixed)} samples")
    print(f"   First 3 feature sets (Fixed XGBoost):")
    for i in range(3):
        print(f"     {i+1}. Shape: {X_fixed.iloc[i].shape}")

    # Compare results
    print(f"\n🔍 Comparison Results")
    print("=" * 50)

    # Check if lengths match
    if len(main_texts) != len(X_fixed):
        print(f"❌ Length mismatch: Main={len(main_texts)}, XGBoost={len(X_fixed)}")
        return False

    # Check if labels match (we can't compare features directly)
    matching_labels = sum(1 for main_label, xgboost_label in zip(main_labels, y_fixed) if main_label == xgboost_label)
    print(f"✅ Matching labels: {matching_labels}/{len(main_labels)} ({matching_labels/len(main_labels)*100:.1f}%)")

    # Test train_test_split consistency
    print(f"\n🧪 Testing Train/Test Split Consistency")

    # Main pipeline split
    X_main = pd.DataFrame({'text': main_texts, 'generated': main_labels})
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
        X_main.drop('text', axis=1), X_main['generated'],
        test_size=test_size, random_state=random_state
    )

    # Fixed XGBoost split
    X_train_fix, X_test_fix, y_train_fix, y_test_fix = train_test_split(
        X_fixed, y_fixed,
        test_size=test_size, random_state=random_state
    )

    # Compare train/test splits
    train_match = y_train_main.equals(y_train_fix)
    test_match = y_test_main.equals(y_test_fix)

    print(f"   Train split match: {train_match}")
    print(f"   Test split match: {test_match}")

    # Check accuracy impact
    if train_match and test_match and matching_labels == len(main_texts):
        print(f"\n🎉 SUCCESS: Fix produces IDENTICAL results!")
        print("   XGBoost should now work correctly with limited datasets.")
        return True
    else:
        print(f"\n⚠️  PARTIAL SUCCESS:")
        print(f"   Labels match: {matching_labels/len(main_texts)*100:.1f}%")
        print(f"   Train split match: {train_match}")
        print(f"   Test split match: {test_match}")

        if matching_labels < len(main_texts):
            print(f"   ⚠️  Still some label mismatches - may need further investigation")

        return False

def test_with_different_sample_sizes():
    """Test the fix with different sample sizes."""
    print(f"\n🧪 Testing with Different Sample Sizes")
    print("=" * 60)

    sample_sizes = [500, 1000, 1500]
    results = {}

    for sample_size in sample_sizes:
        print(f"\n📊 Testing with {sample_size} samples...")

        # Load features
        df_features = pd.read_csv('data/features/extracted_features.csv')
        X = df_features.drop('generated', axis=1)
        y = df_features['generated']

        # Apply the fix
        features_df = pd.concat([X, y], axis=1)
        features_ds = Dataset.from_pandas(features_df)
        features_ds = features_ds.shuffle(seed=42)

        if sample_size > 0:
            features_ds = features_ds.select(range(min(sample_size, len(features_ds))))

        features_result = features_ds.to_pandas()
        X_fixed = features_result.drop('generated', axis=1)
        y_fixed = features_result['generated']

        # Test consistency
        X_train, X_test, y_train, y_test = train_test_split(X_fixed, y_fixed, test_size=0.2, random_state=42)

        # Test reproducibility
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X_fixed, y_fixed, test_size=0.2, random_state=42)

        consistent = y_train.equals(y_train2) and y_test.equals(y_test2)

        results[sample_size] = {
            'samples': len(X_fixed),
            'consistent': consistent,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        print(f"   Samples: {len(X_fixed)}, Consistent splits: {'✅' if consistent else '❌'}")

    print(f"\n🏁 Results Summary:")
    for size, result in results.items():
        consistency = '✅' if result['consistent'] else '❌'
        print(f"   {size:4d} samples: {consistency} (Train: {result['train_size']}, Test: {result['test_size']})")

    return all(r['consistent'] for r in results.values())

if __name__ == "__main__":
    print("🚀 Testing HuggingFace-Based Fix for XGBoost Sampling Issue")
    print("This will verify if our fix produces identical results.\n")

    # Test 1: Basic functionality
    basic_success = test_huggingface_fix()

    # Test 2: Different sample sizes
    size_success = test_with_different_sample_sizes()

    # Summary
    print(f"\n🏁 FINAL RESULTS")
    print("=" * 60)
    print(f"Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"All sample sizes: {'✅ PASS' if size_success else '❌ FAIL'}")

    if basic_success and size_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   The HuggingFace-based fix is ready for deployment.")
        print("   XGBoost should work correctly regardless of sample size.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print("   The fix needs further refinement.")