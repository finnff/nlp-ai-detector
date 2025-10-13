import os
import pandas as pd
import tomllib
import math
from datasets import load_from_disk, Dataset
import numpy as np  # For potential use, though not strictly needed here
import argparse

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

# Map source names to loaders
source_loaders = {
    'ai_text_detection_pile': lambda: (load_from_disk(pile_path)['train'].to_pandas() if 'train' in load_from_disk(pile_path) else load_from_disk(pile_path).to_pandas()),
    'hc3': lambda: (load_from_disk(hc3_path)['train'].to_pandas() if 'train' in load_from_disk(hc3_path) else load_from_disk(hc3_path).to_pandas()),
    'sunilthite': lambda: pd.read_csv(sunilthite_path),
    'daigt_v2': lambda: pd.read_csv(daigt_v2_path),
    'llm_detect_competition': lambda: pd.read_csv(kaggle_comp_path),
    'ah_aitd': lambda: pd.read_excel(ah_aitd_path),
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
    source_dfs['ai_text_detection_pile'] = source_dfs['ai_text_detection_pile'][['text', 'generated']]

# HC3 - Filter and Flatten
if 'hc3' in source_dfs:
    sources_to_include = config['hc3']['sources_to_include']
    source_dfs['hc3'] = source_dfs['hc3'][source_dfs['hc3']['source'].isin(sources_to_include)]
    rows = []
    for _, row in source_dfs['hc3'].iterrows():
        for ans in row['human_answers']:
            if ans:  # Skip empty
                rows.append({'text': ans, 'generated': 0})
        for ans in row['chatgpt_answers']:
            if ans:  # Skip empty
                rows.append({'text': ans, 'generated': 1})
    source_dfs['hc3'] = pd.DataFrame(rows)  # Overwrite with flat df

# sunilthite
if 'sunilthite' in source_dfs:
    source_dfs['sunilthite'] = source_dfs['sunilthite'][['text', 'generated']]

# DAIGT V2
if 'daigt_v2' in source_dfs:
    source_dfs['daigt_v2'] = source_dfs['daigt_v2'][['text', 'label']].rename(columns={'label': 'generated'})

# Kaggle Competition
if 'llm_detect_competition' in source_dfs:
    source_dfs['llm_detect_competition'] = source_dfs['llm_detect_competition'][['text', 'generated']]

# AH&AITD
if 'ah_aitd' in source_dfs:
    source_dfs['ah_aitd']['generated'] = source_dfs['ah_aitd']['label_name'].apply(lambda x: 0 if 'human' in x.lower() else 1)
    source_dfs['ah_aitd'] = source_dfs['ah_aitd'][['text', 'generated']]

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

# Save based on config
save_path = os.path.join(datasets_dir, 'combined_dataset')
if config['use_arrow']:
    Dataset.from_pandas(df).save_to_disk(save_path)
else:
    df.to_csv(f"{save_path}.csv", index=False)

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
