#!/usr/bin/env python3
"""
Test script to validate dataset detection fixes.
Run this on the CUDA server to test the fixes.
"""

import os
import sys

def test_dataset_detection():
    """Test the dataset detection logic."""

    # Import our fixed functions
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess_perplexity_sources import detect_source_datasets

    print("🧪 Testing dataset detection...")

    # Test with the actual datasets directory
    datasets_dir = "data/datasets"

    if not os.path.exists(datasets_dir):
        print(f"❌ Datasets directory not found: {datasets_dir}")
        return False

    sources = detect_source_datasets(datasets_dir)

    print(f"\n📊 Detection Results:")
    print(f"Found {len(sources)} source datasets:")

    for name, config in sources.items():
        print(f"  ✅ {name}:")
        print(f"    Type: {config['type']}")
        print(f"    Input: {config['input_path']}")
        print(f"    Output: {config['output_path']}")
        print(f"    Text column: {config['text_column']}")

    # Check for specific expected sources
    expected_sources = ['hc3', 'ah_aitd', 'daigt_v2', 'sunilthite', 'llm_detect_competition']
    found_sources = set(sources.keys())

    print(f"\n🎯 Expected sources: {expected_sources}")
    print(f"🔍 Found sources: {list(found_sources)}")

    missing = set(expected_sources) - found_sources
    if missing:
        print(f"⚠️  Missing sources: {list(missing)}")
    else:
        print("✅ All expected sources found!")

    # Specifically check for Arrow datasets
    arrow_sources = [name for name, config in sources.items() if config['type'] == 'arrow']
    print(f"\n📊 Arrow datasets found: {arrow_sources}")

    if 'hc3' in arrow_sources:
        print("✅ HC3 Arrow dataset detected!")
    else:
        print("⚠️  HC3 Arrow dataset not detected")

    return True

if __name__ == "__main__":
    test_dataset_detection()