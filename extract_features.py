import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import re


default_dataset = 'data/datasets/combined_dataset.csv'
default_output = 'data/features/extracted_features.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=default_dataset)
parser.add_argument('--output', default=default_output)
args = parser.parse_args()

print(f"Loading dataset from {args.dataset}")
if args.dataset.endswith('.csv'):
    df = pd.read_csv(args.dataset)
else:
   # TODO: Handle arrow format if needed
    raise ValueError(f"Unsupported dataset format: {args.dataset}")

print(f"Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

def extract_lexical_features(text):
    # - Type-token ratio
    # - Unique word ratio
    # - Average word length
    # - Vocabulary richness
    features = {}
    return features


def extract_syntactic_features(text):
    # - Average sentence length
    # - Sentence length variance
    # - Parse tree depth
    # - Clause complexity
    
    features = {}
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    features['average_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
    features['sentence_length_variance'] = np.var(sentence_lengths)
    
    return features

def extract_punctuation_features(text):
    features = {}
    word_count = len(text.split())
    
    punctuation_features = {'comma': ',', 'period': '.', 'exclamation_mark': '!', 'question_mark': '?'}

    for feature, punctuation in punctuation_features.items():
        features[f'{feature}_ratio'] = text.count(punctuation) / word_count
  
    features['punctuation_per_word'] = sum(text.count(punctuation) for punctuation in punctuation_features.values()) / word_count

    return features

def extract_pos_features(text):
    # spacy? nltk?
    # - Noun, adverb, adjective, verb ++ ratios
    features = {}

    return features


def extract_perplexity_features(text):
    features = {}
    return features


def extract_burstiness_features(text):
    features = {}
    return features


def extract_all_features(text):
    features = {}
    
    features.update(extract_lexical_features(text))
    features.update(extract_syntactic_features(text))
    features.update(extract_punctuation_features(text))
    features.update(extract_pos_features(text))
    features.update(extract_perplexity_features(text))
    features.update(extract_burstiness_features(text))
    
    return features


print("\nExtracting features...")
feature_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    text = row['text']
    label = row['generated']
        
    # since most of the features need words to be extracted, or normalized by word count, skip texts with 0 words.
    # (TODO discuss this)
    if pd.isna(text) or not isinstance(text, str) or len(text.split()) == 0:
        # print(f"Skipping row {idx}: {text}")
        continue
    
    features = extract_all_features(text)
    features['generated'] = label
    
    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

print(f"\nSaving features to {args.output}")
os.makedirs(os.path.dirname(args.output), exist_ok=True)
features_df.to_csv(args.output, index=False)

print(f"\nFeature extraction complete")
print(f"Shape: {features_df.shape}")
print(f"Features extracted: {[col for col in features_df.columns]}")
print(f"\nFeature stats:")
print(features_df.describe())

