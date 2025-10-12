import os
import pandas as pd
from datasets import load_from_disk

datasets_dir = 'data/datasets'
dataset_paths = {
    'ai_text_detection_pile': os.path.join(datasets_dir, 'ai_text_detection_pile'),
    'hc3': os.path.join(datasets_dir, 'hc3'),
    'sunilthite': os.path.join(datasets_dir, 'sunilthite', 'sunilthite_Training_Essay_Data.csv'),
    'daigt_v2': os.path.join(datasets_dir, 'daigt_v2', 'DAIGT_v2_train_v2_drcat_02.csv'),
    'llm_detect_competition': os.path.join(datasets_dir, 'llm_detect_competition', 'kaggleComp_train_essays.csv'),
    'ah_aitd': os.path.join(datasets_dir, 'ah_aitd', 'AHAIRD_Dataset.xlsx'),
}

def analyze_dataset(name, path):
    if name in ['ai_text_detection_pile', 'hc3']:
        if not os.path.exists(path):
            print(f"{name}: Dataset not found at {path}")
            return None 
        
        ds = load_from_disk(path)
        if 'train' in ds:
            df = ds['train'].to_pandas()
        else:
            df = ds.to_pandas()
            
    elif name == 'ah_aitd':
        # Excel format
        if not os.path.exists(path):
            print(f"{name}: Dataset not found at {path}")
            return None
        df = pd.read_excel(path)
        
    else:
        # CSV format
        if not os.path.exists(path):
            print(f"{name}: Dataset not found at {path}")
            return None
        df = pd.read_csv(path)
    
    # Standardize the 'generated' column based on dataset
    if name == 'ai_text_detection_pile':
        df['generated'] = df['source'].apply(lambda x: 0 if x == 'human' else 1)
        
    elif name == 'hc3':
        # HC3 needs special handling, flatten first
        rows = []
        for _, row in df.iterrows():
            for ans in row['human_answers']:
                if ans: 
                    rows.append({'generated': 0})
            for ans in row['chatgpt_answers']:
                if ans: 
                    rows.append({'generated': 1})
        df = pd.DataFrame(rows)
        
    elif name == 'daigt_v2':
        df['generated'] = df['label']   
        
    elif name == 'ah_aitd':
        df['generated'] = df['label_name'].apply(lambda x: 0 if 'human' in x.lower() else 1) 
        
    human_count = len(df[df['generated'] == 0])
    ai_count = len(df[df['generated'] == 1])
    total = len(df)
    
    if total == 0:
        print(f"{name}: Empty dataset")
        return None
        
    human_pct = (human_count / total) * 100
    ai_pct = (ai_count / total) * 100
    
    print(f"{name:<25}: {total:>8} total | Human: {human_count:>7} ({human_pct:>5.1f}%) | AI: {ai_count:>7} ({ai_pct:>5.1f}%)")

for name, path in dataset_paths.items():
    analyze_dataset(name, path)
