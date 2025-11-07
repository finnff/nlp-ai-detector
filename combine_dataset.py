import os
import pandas as pd
import tomllib
import math
import datasets
from datasets import load_from_disk, Dataset
import numpy as np  # For potential use, though not strictly needed here
import argparse
import datetime
import json

# Disable caching to force reload datasets
datasets.disable_caching()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='dataset_configuration.toml', help='Path to the TOML configuration file')
args = parser.parse_args()

# Load TOML configuration from file
with open(args.config, 'rb') as f:
    config = tomllib.load(f)

# Set RANDOM_STATE based on config
if config.get('random', False):
    RANDOM_STATE = np.random.randint(1, 101)
else:
    RANDOM_STATE = 0

data_dir = 'data'
datasets_dir = os.path.join(data_dir, 'datasets')

# Define paths
pile_path = os.path.join(datasets_dir, 'ai_text_detection_pile')
hc3_path = os.path.join(datasets_dir, 'hc3')
sunilthite_path = os.path.join(datasets_dir, 'sunilthite', 'sunilthite_Training_Essay_Data.csv')
daigt_v2_path = os.path.join(datasets_dir, 'daigt_v2', 'DAIGT_v2_train_v2_drcat_02.csv')
kaggle_comp_path = os.path.join(datasets_dir, 'llm_detect_competition', 'kaggleComp_train_essays.csv')
ah_aitd_path = os.path.join(datasets_dir, 'ah_aitd', 'AHAIRD_Dataset.xlsx')

def standardize_dataset(df, source_name):
    """Standardize dataset while preserving perplexity and other important columns."""
    # Always keep text and generated
    columns_to_keep = ['text', 'generated']

    # Add perplexity if present
    if 'perplexity' in df.columns:
        columns_to_keep.append('perplexity')
        non_nan = df['perplexity'].notna().sum()
        print(f"   🧠 Preserved perplexity for {non_nan}/{len(df)} samples")

    # Keep only available columns
    available_columns = [col for col in columns_to_keep if col in df.columns]

    if 'perplexity' in available_columns:
        print(f"✅ Standardized {source_name}: kept columns {available_columns}")
    else:
        print(f"📄 Standardized {source_name}: kept columns {available_columns} (no perplexity)")

    return df[available_columns]

def load_with_fallback(perplexity_path, original_path, file_type='arrow'):
    """Load dataset from perplexity-enhanced version if available, otherwise fallback to original."""
    try:
        if file_type == 'arrow':
            if os.path.exists(perplexity_path):
                print(f"🧠 Using perplexity-enhanced dataset: {perplexity_path}")
                dataset = load_from_disk(perplexity_path, keep_in_memory=True)
                return dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()
            elif os.path.exists(original_path):
                print(f"📄 Using original dataset: {original_path}")
                dataset = load_from_disk(original_path, keep_in_memory=True)
                return dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()
        elif file_type == 'csv':
            if os.path.exists(perplexity_path):
                print(f"🧠 Using perplexity-enhanced dataset: {perplexity_path}")
                return pd.read_csv(perplexity_path)
            elif os.path.exists(original_path):
                print(f"📄 Using original dataset: {original_path}")
                return pd.read_csv(original_path)
        elif file_type == 'excel':
            if os.path.exists(perplexity_path):
                print(f"🧠 Using perplexity-enhanced dataset: {perplexity_path}")
                return pd.read_csv(perplexity_path)  # Enhanced version saved as CSV
            elif os.path.exists(original_path):
                print(f"📄 Using original dataset: {original_path}")
                return pd.read_excel(original_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"📄 Falling back to original dataset: {original_path}")
        if file_type == 'arrow':
            dataset = load_from_disk(original_path, keep_in_memory=True)
            return dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()
        elif file_type == 'csv':
            return pd.read_csv(original_path)
        elif file_type == 'excel':
            return pd.read_excel(original_path)

    raise FileNotFoundError(f"Neither perplexity-enhanced nor original dataset found: {perplexity_path}, {original_path}")

# Map source names to loaders with fallback support
source_loaders = {
    'ai_text_detection_pile': lambda: load_with_fallback(
        os.path.join(datasets_dir, 'ai_text_detection_pile', 'dataset_with_perplexity'),
        pile_path,
        'arrow'
    ),
    'hc3': lambda: load_with_fallback(
        os.path.join(datasets_dir, 'hc3', 'dataset_with_perplexity'),
        hc3_path,
        'arrow'
    ),
    'daigt_v2': lambda: load_with_fallback(
        daigt_v2_path.replace('.csv', '_with_perplexity.csv'),
        daigt_v2_path,
        'csv'
    ),
    'sunilthite': lambda: load_with_fallback(
        sunilthite_path.replace('.csv', '_with_perplexity.csv'),
        sunilthite_path,
        'csv'
    ),
    'ah_aitd': lambda: load_with_fallback(
        ah_aitd_path.replace('.xlsx', '_with_perplexity.csv'),
        ah_aitd_path,
        'excel'
    ),
    'llm_detect_competition': lambda: load_with_fallback(
        kaggle_comp_path.replace('.csv', '_with_perplexity.csv'),
        kaggle_comp_path,
        'csv'
    ),
}

# Get enabled sources
enabled_sources = [key for key in source_loaders if config.get(key, {}).get('enabled', False)]

# Load enabled datasets
source_dfs = {}
for key in enabled_sources:
    print(f"Loading {key}...")
    source_dfs[key] = source_loaders[key]()

# Standardize each dataset to 'text' and 'generated' columns

# AI Text Detection Pile
if 'ai_text_detection_pile' in source_dfs:
    source_dfs['ai_text_detection_pile']['generated'] = source_dfs['ai_text_detection_pile']['source'].apply(lambda x: 0 if x == 'human' else 1)
    source_dfs['ai_text_detection_pile'] = standardize_dataset(source_dfs['ai_text_detection_pile'], 'ai_text_detection_pile')

# HC3 - Check if already flattened or needs processing
if 'hc3' in source_dfs:
    df = source_dfs['hc3']

    # Check if dataset is already flattened (has 'text' and 'generated' columns)
    if 'text' in df.columns and 'generated' in df.columns and 'human_answers' not in df.columns:
        print("   HC3 dataset already flattened, using as-is")
        # Already flattened format - just standardize it
        source_dfs['hc3'] = standardize_dataset(df, 'hc3')
    else:
        print("   HC3 dataset needs flattening")
        # Original nested format - apply flattening
        sources_to_include = config['hc3']['sources_to_include']
        df = df[df['source'].isin(sources_to_include)]
        rows = []
        for _, row in df.iterrows():
            for ans in row['human_answers']:
                if ans:  # Skip empty
                    rows.append({'text': ans, 'generated': 0})
            for ans in row['chatgpt_answers']:
                if ans:  # Skip empty
                    rows.append({'text': ans, 'generated': 1})
        source_dfs['hc3'] = pd.DataFrame(rows)  # Overwrite with flat df
        source_dfs['hc3'] = standardize_dataset(source_dfs['hc3'], 'hc3')

# sunilthite
if 'sunilthite' in source_dfs:
    source_dfs['sunilthite'] = standardize_dataset(source_dfs['sunilthite'], 'sunilthite')

# DAIGT V2
if 'daigt_v2' in source_dfs:
    source_dfs['daigt_v2'] = source_dfs['daigt_v2'].rename(columns={'label': 'generated'})
    source_dfs['daigt_v2'] = standardize_dataset(source_dfs['daigt_v2'], 'daigt_v2')

# Kaggle Competition
if 'llm_detect_competition' in source_dfs:
    source_dfs['llm_detect_competition'] = standardize_dataset(source_dfs['llm_detect_competition'], 'llm_detect_competition')

# AH&AITD
if 'ah_aitd' in source_dfs:
    source_dfs['ah_aitd']['generated'] = source_dfs['ah_aitd']['label_name'].apply(lambda x: 0 if 'human' in x.lower() else 1)
    source_dfs['ah_aitd'] = standardize_dataset(source_dfs['ah_aitd'], 'ah_aitd')

# Handle relative portions
portions = {}
sum_portions = sum(config[key]['relative_portion'] for key in enabled_sources if config[key]['relative_portion'] > 0)

if sum_portions > 0:
    if abs(sum_portions - 1.0) > 1e-6:
        raise ValueError("Sum of relative portions must equal 1.0")
    for key in enabled_sources:
        portions[key] = config[key]['relative_portion']
else:
    # Default equal
    equal_portion = 1.0 / len(enabled_sources)
    for key in enabled_sources:
        portions[key] = equal_portion

# Print selected configuration
print("Selected Configuration:")
print(f"Title: {config['Title']}")
print(f"Total samples: \033[94m{config['total_samples']}\033[0m")
print(f"Use arrow: {config['use_arrow']}")
print(f"Random: {config.get('random', False)}")
print(f"Balance classes: {config.get('balance_classes')}")

# Handle total_samples
if config['total_samples'] == 'MAX':
    sampled_dfs = []
    for key in enabled_sources:
        df = source_dfs[key].copy()
        df['origin_source'] = key
        sampled_dfs.append(df)
else:
    total_samples = config['total_samples']
    balance_classes = config.get('balance_classes')
    sampled_dfs = []    
    for i, key in enumerate(enabled_sources):
        portion = portions[key]
        target_n = math.floor(total_samples * portion)
        if sum_portions == 0:
            # Adjust for equal with remainder
            base_n = total_samples // len(enabled_sources)
            remainder = total_samples % len(enabled_sources)
            target_n = base_n + (1 if i < remainder else 0)
        df = source_dfs[key]
        
        if balance_classes:
            human_df = df[df['generated'] == 0]
            ai_df = df[df['generated'] == 1]
            # limited by the smallest class per source
            max_balanced = 2 * min(len(human_df), len(ai_df))
            target_n = min(target_n, max_balanced)
            n_per_class = target_n // 2
            sampled = pd.concat([
                human_df.sample(n=n_per_class, random_state=RANDOM_STATE, replace=False),
                ai_df.sample(n=n_per_class, random_state=RANDOM_STATE, replace=False)
            ], ignore_index=True)
        else:
            n = min(target_n, len(df))
            sampled = df.sample(n=n, random_state=RANDOM_STATE, replace=False)
        
        sampled['origin_source'] = key
        sampled_dfs.append(sampled)

# Combine
df = pd.concat(sampled_dfs, ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Get contributions
contributions = df['origin_source'].value_counts()

# Drop origin_source
df = df.drop(columns=['origin_source'])

# Debug: Show final dataset info
print(f"\n📊 Final combined dataset info:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
if 'perplexity' in df.columns:
    non_nan_ppl = df['perplexity'].notna().sum()
    print(f"   🧠 Perplexity available for {non_nan_ppl}/{len(df)} samples ({non_nan_ppl/len(df)*100:.1f}%)")
else:
    print(f"   ⚠️  No perplexity column found")

# Save based on config first
save_path = os.path.join(datasets_dir, 'combined_dataset')
if config['use_arrow']:
    Dataset.from_pandas(df).save_to_disk(save_path)
else:
    df.to_csv(f"{save_path}.csv", index=False)

# Calculate human/AI counts after processing
generated_counts = df['generated'].value_counts()
human_count = generated_counts.get(0, 0)
ai_count = generated_counts.get(1, 0)

# Save dataset configuration metadata
metadata = {
    'title': config['Title'],
    'total_samples': config['total_samples'],
    'actual_samples': len(df),
    'use_arrow': config['use_arrow'],
    'random': config.get('random', False),
    'balance_classes': config.get('balance_classes', False),
    'enabled_datasets': enabled_sources,
    'test_split': 0.2,  # Default test split used in main.py
    'timestamp': datetime.datetime.now().isoformat(),
    'human_count': int(human_count),
    'ai_count': int(ai_count),
    'has_perplexity': 'perplexity' in df.columns,
    'dataset_shape': df.shape
}

# Save metadata as JSON
metadata_path = os.path.join(datasets_dir, 'dataset_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"💾 Dataset metadata saved: {metadata_path}")

# Print sources with details
print("\nSources:")
total_contrib = len(df)
for key in enabled_sources:
    df_source = source_dfs[key]
    count = len(df_source)
    human_count = len(df_source[df_source['generated'] == 0])
    ai_count = len(df_source[df_source['generated'] == 1])
    contrib = contributions[key]
    perc_of_source = (contrib / count) * 100
    perc_of_combined = (contrib / total_contrib) * 100
    print(f"\033[92m[ENABLED]\033[0m {key:<25}: {count:>7} total (H:{human_count:>6}, AI:{ai_count:>6}) → \033[93m{contrib:>5}\033[0m samples ({perc_of_source:>5.2f}% of source, {perc_of_combined:>5.2f}% of combined)")

# Print required
print("Combined Dataset Head: \n")
print(df.head())

print("\n")
print("Shape:", df.shape)
# Print generated counts
generated_counts = df['generated'].value_counts()
human_count = generated_counts.get(0, 0)
ai_count = generated_counts.get(1, 0)
total = df.shape[0]
human_perc = (human_count / total) * 100
ai_perc = (ai_count / total) * 100
print(f"Human Written: \033[96m{human_count}\033[0m ({human_perc:.2f}%)")
print(f"Ai Generated: \033[95m{ai_count}\033[0m ({ai_perc:.2f}%)")


# =============================================================================
# CROSS-DATASET EVALUATION FUNCTIONS
# =============================================================================

def create_training_dataset(exclude_dataset: str, config_path: str = 'dataset_configuration.toml') -> pd.DataFrame:
    """
    Create training dataset excluding specified dataset for cross-dataset evaluation.

    Args:
        exclude_dataset: Dataset name to exclude from training
        config_path: Path to configuration file

    Returns:
        Combined training DataFrame with all datasets except exclude_dataset
    """
    # Load configuration
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Set random state
    if config.get('random', False):
        RANDOM_STATE = np.random.randint(1, 101)
    else:
        RANDOM_STATE = 0

    print(f"Creating training dataset excluding: {exclude_dataset}")

    # Get source loaders (reuse existing code)
    def load_hc3_dataset():
        """Load HC3 dataset properly handling DatasetDict structure"""
        try:
            dataset_dict = datasets.load_from_disk(hc3_path)
            # Try to get the train split, or use the first available split
            if 'train' in dataset_dict:
                return dataset_dict['train'].to_pandas()
            elif isinstance(dataset_dict, datasets.DatasetDict):
                # Use the first available split
                first_split = list(dataset_dict.keys())[0]
                return dataset_dict[first_split].to_pandas()
            else:
                # It's already a Dataset
                return dataset_dict.to_pandas()
        except Exception as e:
            print(f"Error loading HC3 dataset: {e}")
            return pd.DataFrame()

    source_loaders = {
        'ai_text_detection_pile': lambda: datasets.load_from_disk(pile_path).to_pandas(),
        'hc3': load_hc3_dataset,
        'sunilthite': lambda: pd.read_csv(sunilthite_path),
        'daigt_v2': lambda: pd.read_csv(daigt_v2_path),
        'ah_aitd': lambda: pd.read_excel(ah_aitd_path),
        'llm_detect_competition': lambda: pd.read_csv(kaggle_comp_path),
    }

    # Get enabled sources, excluding the specified dataset
    enabled_sources = [
        key for key in source_loaders
        if config.get(key, {}).get('enabled', False) and key != exclude_dataset
    ]

    print(f"Training datasets: {enabled_sources}")

    if not enabled_sources:
        print(f"⚠️  No datasets enabled after excluding {exclude_dataset}")
        return pd.DataFrame()

    # Load and process enabled datasets
    source_dfs = {}
    for key in enabled_sources:
        print(f"Loading {key}...")
        try:
            source_dfs[key] = source_loaders[key]()
        except Exception as e:
            print(f"⚠️  Could not load {key}: {e}")
            continue

    # Standardize datasets (reuse existing logic)
    # AI Text Detection Pile
    if 'ai_text_detection_pile' in source_dfs:
        source_dfs['ai_text_detection_pile']['generated'] = source_dfs['ai_text_detection_pile']['source'].apply(lambda x: 0 if x == 'human' else 1)
        source_dfs['ai_text_detection_pile'] = standardize_dataset(source_dfs['ai_text_detection_pile'], 'ai_text_detection_pile')

    # HC3
    if 'hc3' in source_dfs:
        df = source_dfs['hc3']
        if 'text' in df.columns and 'generated' in df.columns and 'human_answers' not in df.columns:
            print("   HC3 dataset already flattened, using as-is")
            source_dfs['hc3'] = standardize_dataset(df, 'hc3')
        else:
            print("   HC3 dataset needs flattening")
            sources_to_include = config['hc3']['sources_to_include']
            df = df[df['source'].isin(sources_to_include)]
            rows = []
            for _, row in df.iterrows():
                for ans in row['human_answers']:
                    if ans:
                        rows.append({'text': ans, 'generated': 0})
                for ans in row['chatgpt_answers']:
                    if ans:
                        rows.append({'text': ans, 'generated': 1})
            source_dfs['hc3'] = pd.DataFrame(rows)
            source_dfs['hc3'] = standardize_dataset(source_dfs['hc3'], 'hc3')

    # sunilthite
    if 'sunilthite' in source_dfs:
        source_dfs['sunilthite'] = standardize_dataset(source_dfs['sunilthite'], 'sunilthite')

    # daigt_v2
    if 'daigt_v2' in source_dfs:
        # Convert label to generated for daigt_v2
        if 'label' in source_dfs['daigt_v2'].columns:
            source_dfs['daigt_v2']['generated'] = source_dfs['daigt_v2']['label']
            source_dfs['daigt_v2'] = standardize_dataset(source_dfs['daigt_v2'], 'daigt_v2')
        elif 'generated' not in source_dfs['daigt_v2'].columns:
            print("⚠️  Could not find label or generated column in daigt_v2 dataset, skipping...")
            del source_dfs['daigt_v2']

    # ah_aitd
    if 'ah_aitd' in source_dfs:
        # Convert label_id to generated (1=AI, 0=Human)
        if 'label_id' in source_dfs['ah_aitd'].columns:
            source_dfs['ah_aitd']['generated'] = source_dfs['ah_aitd']['label_id']
            source_dfs['ah_aitd'] = standardize_dataset(source_dfs['ah_aitd'], 'ah_aitd')
        elif 'label_name' in source_dfs['ah_aitd'].columns:
            source_dfs['ah_aitd']['generated'] = source_dfs['ah_aitd']['label_name'].apply(lambda x: 1 if x == 'AI' else 0)
            source_dfs['ah_aitd'] = standardize_dataset(source_dfs['ah_aitd'], 'ah_aitd')
        else:
            print("⚠️  Could not find label column in ah_aitd dataset, skipping...")
            del source_dfs['ah_aitd']

    # llm_detect_competition
    if 'llm_detect_competition' in source_dfs:
        source_dfs['llm_detect_competition']['generated'] = source_dfs['llm_detect_competition']['generated'].apply(lambda x: 1 if x == 1 else 0)
        source_dfs['llm_detect_competition'] = standardize_dataset(source_dfs['llm_detect_competition'], 'llm_detect_competition')

    # Combine all training datasets
    sampled_dfs = []
    total_samples = sum(len(df) for df in source_dfs.values())

    for key, df in source_dfs.items():
        # For training, use all available data from each dataset
        sampled = df.copy()
        sampled['origin_source'] = key
        sampled_dfs.append(sampled)

    # Combine and shuffle
    if sampled_dfs:
        df = pd.concat(sampled_dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        # Apply class balancing if specified
        if config.get('balance_classes', False):
            df = balance_dataset_classes(df, RANDOM_STATE)

        # Drop origin_source for training (not needed for model)
        df = df.drop(columns=['origin_source'])

        print(f"✅ Training dataset created: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        if 'perplexity' in df.columns:
            non_nan_ppl = df['perplexity'].notna().sum()
            print(f"   🧠 Perplexity available for {non_nan_ppl}/{len(df)} samples ({non_nan_ppl/len(df)*100:.1f}%)")

        return df
    else:
        print("⚠️  No data loaded for training dataset")
        return pd.DataFrame()


def load_individual_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load a single dataset for testing in cross-dataset evaluation.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        DataFrame with the loaded dataset, standardized format
    """
    print(f"Loading individual dataset: {dataset_name}")

    # Define source loaders
    def load_hc3_dataset():
        """Load HC3 dataset properly handling DatasetDict structure"""
        try:
            dataset_dict = datasets.load_from_disk(hc3_path)
            # Try to get the train split, or use the first available split
            if 'train' in dataset_dict:
                return dataset_dict['train'].to_pandas()
            elif isinstance(dataset_dict, datasets.DatasetDict):
                # Use the first available split
                first_split = list(dataset_dict.keys())[0]
                return dataset_dict[first_split].to_pandas()
            else:
                # It's already a Dataset
                return dataset_dict.to_pandas()
        except Exception as e:
            print(f"Error loading HC3 dataset: {e}")
            return pd.DataFrame()

    source_loaders = {
        'hc3': load_hc3_dataset,
        'sunilthite': lambda: pd.read_csv(sunilthite_path),
        'daigt_v2': lambda: pd.read_csv(daigt_v2_path),
        'ah_aitd': lambda: pd.read_excel(ah_aitd_path),
        'ai_text_detection_pile': lambda: datasets.load_from_disk(pile_path).to_pandas(),
        'llm_detect_competition': lambda: pd.read_csv(kaggle_comp_path),
    }

    if dataset_name not in source_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    try:
        # Load the dataset
        df = source_loaders[dataset_name]()
        print(f"   Loaded {len(df)} samples from {dataset_name}")

        # Standardize the dataset
        if dataset_name == 'hc3':
            # Handle HC3 flattening
            if 'text' in df.columns and 'generated' in df.columns and 'human_answers' not in df.columns:
                print("   HC3 dataset already flattened, using as-is")
                df = standardize_dataset(df, 'hc3')
            else:
                print("   HC3 dataset needs flattening")
                rows = []
                for _, row in df.iterrows():
                    for ans in row['human_answers']:
                        if ans:
                            rows.append({'text': ans, 'generated': 0})
                    for ans in row['chatgpt_answers']:
                        if ans:
                            rows.append({'text': ans, 'generated': 1})
                df = pd.DataFrame(rows)
                df = standardize_dataset(df, 'hc3')

        elif dataset_name == 'ah_aitd':
            # Convert label_id to generated (1=AI, 0=Human)
            if 'label_id' in df.columns:
                df['generated'] = df['label_id']
            elif 'label_name' in df.columns:
                df['generated'] = df['label_name'].apply(lambda x: 1 if x == 'AI' else 0)
            else:
                raise ValueError(f"Could not find label column in ah_aitd dataset")
            df = standardize_dataset(df, 'ah_aitd')

        elif dataset_name == 'ai_text_detection_pile':
            # Convert source to generated labels
            df['generated'] = df['source'].apply(lambda x: 0 if x == 'human' else 1)
            df = standardize_dataset(df, 'ai_text_detection_pile')

        elif dataset_name == 'llm_detect_competition':
            df['generated'] = df['generated'].apply(lambda x: 1 if x == 1 else 0)
            df = standardize_dataset(df, 'llm_detect_competition')

        elif dataset_name == 'daigt_v2':
            # Convert label to generated for daigt_v2
            if 'label' in df.columns:
                df['generated'] = df['label']
                df = standardize_dataset(df, 'daigt_v2')
            else:
                print("⚠️  Could not find label column in daigt_v2 dataset")
                return pd.DataFrame()
        else:
            # Standard datasets (sunilthite)
            df = standardize_dataset(df, dataset_name)

        print(f"   ✅ Standardized {dataset_name}: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        # Show class distribution
        human_count = len(df[df['generated'] == 0])
        ai_count = len(df[df['generated'] == 1])
        print(f"   Human: {human_count}, AI: {ai_count}")

        return df

    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def balance_dataset_classes(df: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """
    Balance dataset classes by downsampling the majority class.

    Args:
        df: Input DataFrame
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    # Separate classes
    human_df = df[df['generated'] == 0]
    ai_df = df[df['generated'] == 1]

    min_class_size = min(len(human_df), len(ai_df))

    if len(human_df) != min_class_size:
        human_df = human_df.sample(n=min_class_size, random_state=random_state)
        print(f"   Downsampled human class from {len(df[df['generated'] == 0])} to {min_class_size}")

    if len(ai_df) != min_class_size:
        ai_df = ai_df.sample(n=min_class_size, random_state=random_state)
        print(f"   Downsampled AI class from {len(df[df['generated'] == 1])} to {min_class_size}")

    balanced_df = pd.concat([human_df, ai_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"   ✅ Balanced dataset: {balanced_df.shape}")
    return balanced_df


if __name__ == "__main__":
    # Allow running combine_dataset.py directly for testing
    pass
